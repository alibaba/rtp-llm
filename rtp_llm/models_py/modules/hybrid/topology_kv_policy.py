"""Topology-aware sparse KV candidate policy.

The policy is an opt-in post-processing layer for indexer top-k results. Learned
`topk_indices` must already be absolute token indices in the downstream sparse
attention coordinate system. For ragged and CP-prefill paths,
`topk_indices_offset` defines the valid absolute interval for each row:
`[offset, offset + length)`. `row_starts` scopes the logits read before top-k
selection and is validated only as a row-count alignment guard; it is not added
to returned top-k coordinates. Negative values are padding. Non-padding learned
indices outside that interval fail fast because silently dropping them can hide
a coordinate-system mismatch.

`stable_scaffold` and `output_contract` describe repeated prompt structure such
as system instructions, tool schemas, and response contracts. Their fingerprint
is only an observability key for repeated stable regions; this module does not
physically compress the KV cache. Counters report selected-token shape and how
many stable tokens were represented by that fingerprint but not selected as raw
top-k candidates.

Merge modes intentionally change the sparse attention candidate set: structural
tokens are admitted within a configured budget fraction and can reorder or evict
learned top-k entries. `learned_kept_tokens` and `learned_evicted_tokens` expose
that semantic change to callers.
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
CoordinateMismatchAction = Literal["raise", "fallback_disabled"]
SUPPORTED_TOPOLOGY_KV_POLICIES = frozenset(
    {
        "disabled",
        "topology_sparse_merge",
        "topology_compress_sparse",
        "topology_only",
    }
)


def normalize_topology_kv_policy(policy: str) -> TopologyKvPolicy:
    normalized = policy.strip().lower()
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
    max_policy_tokens: int = 8192
    max_structural_fraction: float = 0.5
    coordinate_mismatch_action: CoordinateMismatchAction = "raise"

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy", normalize_topology_kv_policy(self.policy))
        if self.coordinate_mismatch_action not in {"raise", "fallback_disabled"}:
            raise ValueError(
                "coordinate_mismatch_action must be 'raise' or 'fallback_disabled'"
            )
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.sink_blocks < 0:
            raise ValueError("sink_blocks must be non-negative")
        if self.local_blocks < 0:
            raise ValueError("local_blocks must be non-negative")
        if self.witness_blocks < 0:
            raise ValueError("witness_blocks must be non-negative")
        if self.max_policy_tokens < 0:
            raise ValueError("max_policy_tokens must be non-negative")
        if not 0.0 < self.max_structural_fraction <= 1.0:
            raise ValueError("max_structural_fraction must be in (0, 1]")


@dataclass(frozen=True)
class TopologyKvCounters:
    raw_selected_tokens: int
    compressed_tokens_represented: int
    unselected_stable_tokens: int
    learned_kept_tokens: int
    learned_evicted_tokens: int
    compression_hits: int
    coordinate_mismatch_fallbacks: int
    policy_bypassed: int
    schedule_ms: float


@dataclass(frozen=True)
class TopologyKvPolicyResult:
    topk_indices: torch.Tensor
    counters: TopologyKvCounters
    stable_fingerprint: str


@dataclass(frozen=True)
class _MergeRowResult:
    values: list[int]
    learned_kept_tokens: int
    learned_evicted_tokens: int


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
    budget: int,
) -> None:
    start = block_id * block_size
    end = min(start + block_size, length)
    for token in range(start, end):
        if token not in seen:
            selected.append(token)
            seen.add(token)
            if len(selected) == budget:
                return


def _structural_tokens(
    length: int,
    config: TopologyKvPolicyConfig,
    drift_row: Optional[torch.Tensor],
    budget: int,
) -> list[int]:
    if length <= 0 or budget <= 0:
        return []
    block_count = (length + config.block_size - 1) // config.block_size
    selected: list[int] = []
    seen: set[int] = set()
    ordered_blocks: list[int] = []
    ordered_seen: set[int] = set()

    def add_block(block_id: int) -> None:
        if block_id not in ordered_seen:
            ordered_blocks.append(block_id)
            ordered_seen.add(block_id)

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
            if len(selected) == budget:
                return selected

    for block_id in ordered_blocks:
        _append_block_tokens(
            selected, seen, block_id, config.block_size, length, budget
        )
        if len(selected) == budget:
            return selected

    return selected


def _merge_row(
    learned_values: list[int],
    structural: list[int],
    local_start: int,
    local_length: int,
    max_structural_fraction: float,
) -> _MergeRowResult:
    budget = len(learned_values)
    selected: list[int] = []
    seen: set[int] = set()
    local_end = local_start + local_length
    structural_budget = (
        budget
        if budget == 0
        else max(1, min(budget, int(budget * max_structural_fraction)))
    )

    for token in structural:
        if local_start <= token < local_end and token not in seen:
            selected.append(token)
            seen.add(token)
            if len(selected) == structural_budget:
                break

    unique_valid_learned = {
        value
        for value in learned_values
        if value >= 0 and local_start <= value < local_end
    }
    for value in learned_values:
        if value < 0:
            continue
        if not local_start <= value < local_end:
            continue
        if value not in seen:
            selected.append(value)
            seen.add(value)
            if len(selected) == budget:
                break

    learned_kept = sum(1 for value in unique_valid_learned if value in selected)
    learned_evicted = max(0, len(unique_valid_learned) - learned_kept)

    while len(selected) < budget:
        selected.append(-1)
    return _MergeRowResult(selected, learned_kept, learned_evicted)


def _validate_learned_row_coordinates(
    learned_values: list[int],
    *,
    row_idx: int,
    local_start: int,
    local_length: int,
) -> None:
    local_end = local_start + local_length
    for value in learned_values:
        if value < 0:
            continue
        if local_start <= value < local_end:
            continue
        raise ValueError(
            "learned topk index outside valid topology KV range: "
            f"row={row_idx}, value={value}, expected absolute index in "
            f"[{local_start}, {local_end})"
        )


def _validate_optional_row_tensor(
    value: Optional[torch.Tensor],
    *,
    name: str,
    rows: int,
) -> None:
    if value is None:
        return
    if value.ndim != 1 or value.numel() != rows:
        raise ValueError(f"{name} must have shape [rows]")


def _zero_counters(topk_indices: torch.Tensor) -> TopologyKvCounters:
    return TopologyKvCounters(
        raw_selected_tokens=int((topk_indices >= 0).sum().item()),
        compressed_tokens_represented=0,
        unselected_stable_tokens=0,
        learned_kept_tokens=0,
        learned_evicted_tokens=0,
        compression_hits=0,
        coordinate_mismatch_fallbacks=0,
        policy_bypassed=0,
        schedule_ms=0.0,
    )


def _passthrough_result(
    topk_indices: torch.Tensor,
    fingerprint: str,
    *,
    coordinate_mismatch_fallbacks: int = 0,
    policy_bypassed: int = 0,
) -> TopologyKvPolicyResult:
    counters = _zero_counters(topk_indices)
    counters = TopologyKvCounters(
        raw_selected_tokens=counters.raw_selected_tokens,
        compressed_tokens_represented=0,
        unselected_stable_tokens=0,
        learned_kept_tokens=0,
        learned_evicted_tokens=0,
        compression_hits=0,
        coordinate_mismatch_fallbacks=coordinate_mismatch_fallbacks,
        policy_bypassed=policy_bypassed,
        schedule_ms=0.0,
    )
    return TopologyKvPolicyResult(topk_indices, counters, fingerprint)


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
    the input `topk_indices`. `topk_indices_offset` derives the row-local valid
    interval. `row_starts` is validated only as a row-count alignment guard and
    is not added to returned coordinates.
    """

    started = time.perf_counter()
    fingerprint = _stable_fingerprint(stable_scaffold, output_contract)

    if config.policy == "disabled":
        return _passthrough_result(topk_indices, fingerprint)
    if topk_indices.ndim != 2:
        raise ValueError("topk_indices must have shape [rows, topk]")
    if lengths.ndim != 1 or lengths.numel() != topk_indices.size(0):
        raise ValueError("lengths must have shape [rows]")
    _validate_optional_row_tensor(row_starts, name="row_starts", rows=lengths.numel())
    _validate_optional_row_tensor(
        topk_indices_offset, name="topk_indices_offset", rows=lengths.numel()
    )

    lengths_cpu = lengths.detach().cpu()
    total_policy_tokens = int(lengths_cpu.sum().item())
    if total_policy_tokens > config.max_policy_tokens:
        return _passthrough_result(topk_indices, fingerprint, policy_bypassed=1)

    topk_cpu = topk_indices.detach().cpu()
    offsets_cpu = (
        topk_indices_offset.detach().cpu()
        if topk_indices_offset is not None
        else torch.zeros_like(lengths_cpu)
    )
    raw_selected_before = int((topk_cpu >= 0).sum().item())
    topk_values = [[int(value) for value in row] for row in topk_cpu.tolist()]
    length_values = [int(value) for value in lengths_cpu.tolist()]
    offset_values = [int(value) for value in offsets_cpu.tolist()]
    rows: list[list[int]] = []
    learned_kept_tokens = 0
    learned_evicted_tokens = 0

    compressed_tokens = 0
    compression_hits = 0
    if config.policy == "topology_compress_sparse" and fingerprint:
        stable_tokens = total_policy_tokens
        compressed_tokens = max(0, stable_tokens - raw_selected_before)
        if previous_fingerprint and previous_fingerprint == fingerprint:
            compression_hits = 1

    for row_idx, (row_length, offset) in enumerate(zip(length_values, offset_values)):
        drift_row = (
            block_drift_scores[row_idx]
            if block_drift_scores is not None and block_drift_scores.ndim == 2
            else None
        )
        local_start = offset
        structural = _structural_tokens(
            row_length, config, drift_row, topk_indices.size(1)
        )
        structural = [token + local_start for token in structural]
        learned_values = (
            [-1] * topk_indices.size(1)
            if config.policy == "topology_only"
            else topk_values[row_idx]
        )
        try:
            _validate_learned_row_coordinates(
                learned_values,
                row_idx=row_idx,
                local_start=local_start,
                local_length=row_length,
            )
        except ValueError:
            if config.coordinate_mismatch_action == "fallback_disabled":
                return _passthrough_result(
                    topk_indices, fingerprint, coordinate_mismatch_fallbacks=1
                )
            raise
        row_result = _merge_row(
            learned_values,
            structural,
            local_start,
            row_length,
            config.max_structural_fraction,
        )
        rows.append(row_result.values)
        learned_kept_tokens += row_result.learned_kept_tokens
        learned_evicted_tokens += row_result.learned_evicted_tokens

    merged = torch.tensor(rows, dtype=topk_indices.dtype, device=topk_indices.device)
    raw_selected_after = sum(1 for row in rows for value in row if value >= 0)
    counters = TopologyKvCounters(
        raw_selected_tokens=raw_selected_after,
        compressed_tokens_represented=compressed_tokens,
        unselected_stable_tokens=compressed_tokens,
        learned_kept_tokens=learned_kept_tokens,
        learned_evicted_tokens=learned_evicted_tokens,
        compression_hits=compression_hits,
        coordinate_mismatch_fallbacks=0,
        policy_bypassed=0,
        schedule_ms=(time.perf_counter() - started) * 1000,
    )
    return TopologyKvPolicyResult(merged, counters, fingerprint)
