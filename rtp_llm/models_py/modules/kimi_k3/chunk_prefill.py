"""Whole-model chunk Prefill planning and round-input construction for Kimi K3.

Round planning and packed-input construction live in the Python model; the
C++ executor observes each planned round through a
``mtp_chunk_prefill_round_hook`` callback and only assembles the mirrored
draft-model input.
"""

from __future__ import annotations

from contextlib import nullcontext

import copy
import json
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Sequence

import torch
from rtp_llm.ops import compute_ops as _trace_ops
from rtp_llm.ops.compute_ops import (
    PyAttentionInputs,
    PyEmbeddingInputs,
    PyModelInputs,
    PyModelOutputs,
    PyMultimodalInputs,
)

from rtp_llm.models_py.distributed.collective_torch import Group, barrier
from rtp_llm.models_py.modules.kimi_k3.kda.prefill import KimiKDACurrentStateRegistry

if TYPE_CHECKING:
    from rtp_llm.ops.compute_ops import PyCacheStorePublishPlan


@dataclass(frozen=True)
class KimiK3ChunkSlice:
    original_batch_idx: int
    source_start: int
    source_end: int
    prefix_length: int
    processed_length: int
    new_length: int
    absolute_start: int
    absolute_end: int
    terminal: bool


@dataclass(frozen=True)
class KimiK3ChunkRound:
    slices: tuple[KimiK3ChunkSlice, ...]

    @property
    def token_count(self) -> int:
        return sum(item.new_length for item in self.slices)


@dataclass(frozen=True)
class KimiK3RowSelection:
    """CPU-only row description, retained in the complete batch plan."""

    ranges: tuple[tuple[int, int], ...]
    token_count: int

    def prepare(self, device):
        index = None
        if len(self.ranges) > 1:
            index = torch.cat(
                [
                    torch.arange(start, start + length, device=device)
                    for start, length in self.ranges
                ]
            )
        return KimiK3PreparedRowSelection(self.ranges, self.token_count, index)


@dataclass(frozen=True)
class KimiK3PreparedRowSelection:
    """Current-round device index; never retained by the CPU batch plan."""

    ranges: tuple[tuple[int, int], ...]
    token_count: int
    index: Optional[torch.Tensor]

    def select(self, hidden):
        if not self.ranges:
            return hidden.narrow(0, 0, 0)
        if len(self.ranges) == 1:
            return hidden.narrow(0, *self.ranges[0])
        if self.index is None:
            raise RuntimeError("draft row selection was not prepared")
        return hidden.index_select(0, self.index)


@dataclass(frozen=True)
class KimiK3BatchPlan:
    rounds: tuple[KimiK3ChunkRound, ...]
    draft_rows: tuple[KimiK3RowSelection, ...]
    terminal_rows: tuple[tuple[tuple[int, int], ...], ...]

    @classmethod
    def from_rounds(cls, rounds):
        selections, terminals = [], []
        for round_plan in rounds:
            ranges, terminal, offset = [], [], 0
            for item in round_plan.slices:
                length = item.new_length - int(item.terminal)
                if length:
                    if ranges and sum(ranges[-1]) == offset:
                        ranges[-1] = (ranges[-1][0], ranges[-1][1] + length)
                    else:
                        ranges.append((offset, length))
                offset += item.new_length
                if item.terminal:
                    terminal.append((item.original_batch_idx, offset - 1))
            selections.append(
                KimiK3RowSelection(tuple(ranges), sum(n for _, n in ranges))
            )
            terminals.append(tuple(terminal))
        return cls(tuple(rounds), tuple(selections), tuple(terminals))



def logical_chunk_round(
    round_plan: KimiK3ChunkRound,
    logical_request_count: int,
) -> KimiK3ChunkRound:
    """Return the real-request prefix of a physically padded Prefill round."""

    logical_slices = tuple(
        item
        for item in round_plan.slices
        if int(item.original_batch_idx) < logical_request_count
    )
    if len(logical_slices) == len(round_plan.slices):
        return round_plan
    return KimiK3ChunkRound(logical_slices)


@dataclass(frozen=True)
class KimiK3ChunkRdmaPublishStep:
    """One original-batch publication frontier update."""

    begin_blocks: tuple[int, ...]
    end_blocks: tuple[int, ...]
    terminal: tuple[bool, ...]

    @property
    def has_full_blocks(self) -> bool:
        return any(
            begin < end for begin, end in zip(self.begin_blocks, self.end_blocks)
        )

    @property
    def terminal_indices(self) -> tuple[int, ...]:
        return tuple(index for index, terminal in enumerate(self.terminal) if terminal)

    def to_op_plan(self) -> PyCacheStorePublishPlan:
        from rtp_llm.ops.compute_ops import PyCacheStorePublishPlan

        plan = PyCacheStorePublishPlan()
        plan.begin_block_host = torch.tensor(self.begin_blocks, dtype=torch.int32)
        plan.end_block_host = torch.tensor(self.end_blocks, dtype=torch.int32)
        plan.terminal_host = torch.tensor(self.terminal, dtype=torch.bool)
        return plan


class KimiK3ChunkRdmaPublisher:
    """Track monotonic FULL frontiers and terminal-only KDA publications."""

    def __init__(
        self,
        input_lengths: Sequence[int],
        prefix_lengths: Sequence[int],
        *,
        transfer_page_tokens: int,
        kda_layer_indices: Sequence[int],
    ) -> None:
        self.input_lengths = tuple(int(value) for value in input_lengths)
        self.prefix_lengths = tuple(int(value) for value in prefix_lengths)
        if (
            len(self.input_lengths) != len(self.prefix_lengths)
            or not self.input_lengths
        ):
            raise ValueError(
                "K3 chunk RDMA publisher requires matching non-empty lengths"
            )
        if transfer_page_tokens <= 0:
            raise ValueError(
                "K3 chunk RDMA transfer page size must be positive, "
                f"got {transfer_page_tokens}"
            )
        self.transfer_page_tokens = int(transfer_page_tokens)
        self.kda_layer_indices = frozenset(
            int(value) for value in kda_layer_indices
        )
        self._frontier = [0] * len(self.input_lengths)
        self._terminal = [False] * len(self.input_lengths)
        self._published_kda: set[tuple[int, int]] = set()
        self._final_pages = tuple(
            (prefix + length + self.transfer_page_tokens - 1)
            // self.transfer_page_tokens
            for prefix, length in zip(self.prefix_lengths, self.input_lengths)
        )

    @property
    def frontier(self) -> tuple[int, ...]:
        return tuple(self._frontier)

    def _make_step(
        self,
        end_blocks: Sequence[int],
        terminal: Sequence[bool],
    ) -> KimiK3ChunkRdmaPublishStep:
        begins = tuple(self._frontier)
        ends = tuple(int(value) for value in end_blocks)
        terminals = tuple(bool(value) for value in terminal)
        if len(ends) != len(begins) or len(terminals) != len(begins):
            raise RuntimeError("K3 chunk RDMA publication batch size changed")
        for index, (begin, end, final) in enumerate(
            zip(begins, ends, self._final_pages)
        ):
            if end < begin or end > final:
                raise RuntimeError(
                    "K3 chunk RDMA publication frontier is non-monotonic or out of range: "
                    f"request={index} begin={begin} end={end} final={final}"
                )
        return KimiK3ChunkRdmaPublishStep(
            begin_blocks=begins,
            end_blocks=ends,
            terminal=terminals,
        )

    def prefix_step(self) -> KimiK3ChunkRdmaPublishStep:
        return self._make_step(
            [prefix // self.transfer_page_tokens for prefix in self.prefix_lengths],
            [False] * len(self.input_lengths),
        )

    def round_step(self, round_plan: KimiK3ChunkRound) -> KimiK3ChunkRdmaPublishStep:
        ends = list(self._frontier)
        terminals = [False] * len(self.input_lengths)
        for item in round_plan.slices:
            index = int(item.original_batch_idx)
            if index < 0 or index >= len(self.input_lengths):
                raise RuntimeError(
                    f"K3 chunk RDMA round has invalid request index {index}"
                )
            if self._terminal[index]:
                raise RuntimeError(
                    f"K3 chunk RDMA request {index} appeared after its terminal round"
                )
            expected_frontier = int(item.absolute_start) // self.transfer_page_tokens
            if self._frontier[index] != expected_frontier:
                raise RuntimeError(
                    "K3 chunk RDMA publication frontier does not match the round start: "
                    f"request={index} frontier={self._frontier[index]} "
                    f"expected={expected_frontier}"
                )
            absolute_end = int(item.absolute_end)
            if item.terminal:
                end = (
                    absolute_end + self.transfer_page_tokens - 1
                ) // self.transfer_page_tokens
            else:
                end = absolute_end // self.transfer_page_tokens
            ends[index] = end
            terminals[index] = bool(item.terminal)
        return self._make_step(ends, terminals)

    def commit(self, step: KimiK3ChunkRdmaPublishStep) -> None:
        if tuple(self._frontier) != step.begin_blocks:
            raise RuntimeError(
                "K3 chunk RDMA publication committed against a stale frontier: "
                f"actual={tuple(self._frontier)} expected={step.begin_blocks}"
            )
        self._frontier[:] = step.end_blocks
        for index in step.terminal_indices:
            if self._terminal[index]:
                raise RuntimeError(
                    f"K3 chunk RDMA request {index} was committed terminal twice"
                )
            self._terminal[index] = True

    def record_kda_layer(
        self, layer_idx: int, step: KimiK3ChunkRdmaPublishStep
    ) -> None:
        layer_idx = int(layer_idx)
        for request_idx in step.terminal_indices:
            key = (request_idx, layer_idx)
            if key in self._published_kda:
                raise RuntimeError(
                    "K3 chunk RDMA KDA state was published twice: "
                    f"request={request_idx} layer={layer_idx}"
                )
            self._published_kda.add(key)

    def validate_complete(self) -> None:
        if not all(self._terminal):
            missing = [index for index, value in enumerate(self._terminal) if not value]
            raise RuntimeError(
                f"K3 chunk RDMA requests did not reach terminal publication: {missing}"
            )
        if tuple(self._frontier) != self._final_pages:
            raise RuntimeError(
                "K3 chunk RDMA final FULL frontiers are incomplete: "
                f"actual={tuple(self._frontier)} expected={self._final_pages}"
            )
        expected_kda = {
            (request_idx, layer_idx)
            for request_idx in range(len(self.input_lengths))
            for layer_idx in self.kda_layer_indices
        }
        if self._published_kda != expected_kda:
            missing = sorted(expected_kda - self._published_kda)
            unexpected = sorted(self._published_kda - expected_kda)
            raise RuntimeError(
                "K3 chunk RDMA KDA publications are incomplete or duplicated: "
                f"missing={missing} unexpected={unexpected}"
            )


@dataclass(frozen=True)
class KimiK3ChunkPublishContext:
    """Publish one chunk round immediately after each layer finishes."""

    writer: Any
    publisher: KimiK3ChunkRdmaPublisher
    step: KimiK3ChunkRdmaPublishStep
    op_plan: Any

    def publish_layer(self, layer_idx: int, layer: Any, layer_cache: Any) -> None:
        if layer.is_kda:
            if not self.step.terminal_indices:
                return
            layer.prepare_kda_cache_store(layer_cache)
        elif not self.step.has_full_blocks:
            return

        self.writer(layer_cache, self.op_plan)
        if layer.is_kda:
            self.publisher.record_kda_layer(layer_idx, self.step)


class KimiK3ChunkCachePublisher:
    """Own the optional chunk-wise CacheStore publication lifecycle."""

    def __init__(
        self,
        *,
        writer: Any = None,
        publisher: Optional[KimiK3ChunkRdmaPublisher] = None,
        layers: Sequence[Any] = (),
        kv_cache: Any = None,
    ) -> None:
        self._writer = writer
        self._publisher = publisher
        self._layers = layers
        self._kv_cache = kv_cache

    @classmethod
    def create(
        cls,
        attention_inputs: PyAttentionInputs,
        kv_cache: Any,
        layers: Sequence[Any],
        *,
        input_lengths: Sequence[int],
        prefix_lengths: Sequence[int],
        transfer_page_tokens: int,
    ) -> KimiK3ChunkCachePublisher:
        if not chunkwise_rdma_enabled():
            return cls()

        from rtp_llm.models_py.modules.base.common.kvcache_store import (
            create_write_cache_store_impl,
        )

        writer = create_write_cache_store_impl(attention_inputs, kv_cache)
        if writer is None:
            raise RuntimeError(
                "K3 chunk-wise RDMA requires original-batch CacheStore metadata"
            )
        return cls(
            writer=writer,
            publisher=KimiK3ChunkRdmaPublisher(
                input_lengths,
                prefix_lengths,
                transfer_page_tokens=transfer_page_tokens,
                kda_layer_indices=(
                    layer_idx for layer_idx, layer in enumerate(layers) if layer.is_kda
                ),
            ),
            layers=layers,
            kv_cache=kv_cache,
        )

    @property
    def enabled(self) -> bool:
        return self._publisher is not None

    def _context(self, step: KimiK3ChunkRdmaPublishStep) -> KimiK3ChunkPublishContext:
        return KimiK3ChunkPublishContext(
            writer=self._writer,
            publisher=self._publisher,
            step=step,
            op_plan=step.to_op_plan(),
        )

    def publish_prefix(self) -> None:
        if self._publisher is None:
            return
        step = self._publisher.prefix_step()
        if step.has_full_blocks:
            context = self._context(step)
            for layer_idx, layer in enumerate(self._layers):
                context.publish_layer(
                    layer_idx,
                    layer,
                    self._kv_cache.get_layer_cache(layer_idx),
                )
        self._publisher.commit(step)

    def begin_round(
        self, round_plan: KimiK3ChunkRound
    ) -> Optional[KimiK3ChunkPublishContext]:
        if self._publisher is None:
            return None
        return self._context(self._publisher.round_step(round_plan))

    def commit_round(self, context: Optional[KimiK3ChunkPublishContext]) -> None:
        if context is not None:
            context.publisher.commit(context.step)

    def validate_complete(self) -> None:
        if self._publisher is not None:
            self._publisher.validate_complete()


def chunkwise_rdma_enabled() -> bool:
    """Return whether K3 chunk-wise cache publication is enabled."""

    raw = os.environ.get("KIMI_K3_CHUNKWISE_RDMA", "0")
    if raw not in ("0", "1"):
        raise RuntimeError("KIMI_K3_CHUNKWISE_RDMA must be 0 or 1")
    return raw == "1"


def _source_offsets(lengths: Sequence[int]) -> list[int]:
    offsets = [0]
    for length in lengths:
        if int(length) <= 0:
            raise ValueError(f"K3 chunk input lengths must be positive, got {length}")
        offsets.append(offsets[-1] + int(length))
    return offsets


def plan_kimi_k3_chunk_rounds(
    input_lengths: Sequence[int],
    prefix_lengths: Sequence[int],
    *,
    chunk_budget: int,
    alignment_tokens: int,
) -> tuple[KimiK3ChunkRound, ...]:
    """Split a packed Prefill batch at absolute joint-checkpoint boundaries."""

    lengths = [int(value) for value in input_lengths]
    prefixes = [int(value) for value in prefix_lengths]
    if len(lengths) != len(prefixes):
        raise ValueError(
            "K3 chunk input/prefix batch sizes differ: "
            f"input={len(lengths)} prefix={len(prefixes)}"
        )
    if not lengths:
        raise ValueError("K3 chunk planner requires at least one request")
    if chunk_budget <= 0:
        raise ValueError(f"K3 chunk budget must be positive, got {chunk_budget}")
    if alignment_tokens <= 0:
        raise ValueError(
            f"K3 chunk checkpoint alignment must be positive, got {alignment_tokens}"
        )
    if any(prefix < 0 for prefix in prefixes):
        raise ValueError(f"K3 prefix lengths must be non-negative, got {prefixes}")
    if any(prefix % alignment_tokens for prefix in prefixes):
        raise ValueError(
            "whole-model K3 chunk Prefill requires checkpoint-aligned prefixes: "
            f"prefixes={prefixes} alignment={alignment_tokens}"
        )
    if chunk_budget < alignment_tokens and any(
        length > chunk_budget for length in lengths
    ):
        raise ValueError(
            "K3 chunk budget must reach one joint checkpoint when a request "
            f"spans rounds: budget={chunk_budget}, checkpoint={alignment_tokens}"
        )

    source_offsets = _source_offsets(lengths)
    processed = [0] * len(lengths)
    rounds: list[KimiK3ChunkRound] = []
    while any(done < total for done, total in zip(processed, lengths)):
        available = chunk_budget
        round_slices: list[KimiK3ChunkSlice] = []
        for request_idx, total_length in enumerate(lengths):
            done = processed[request_idx]
            remaining = total_length - done
            if remaining <= 0 or available <= 0:
                continue

            terminal = remaining <= available
            if terminal:
                take = remaining
            else:
                absolute_start = prefixes[request_idx] + done
                aligned_end = (
                    (absolute_start + available) // alignment_tokens
                ) * alignment_tokens
                take = aligned_end - absolute_start
                if take <= 0:
                    continue

            absolute_start = prefixes[request_idx] + done
            absolute_end = absolute_start + take
            source_start = source_offsets[request_idx] + done
            round_slices.append(
                KimiK3ChunkSlice(
                    original_batch_idx=request_idx,
                    source_start=source_start,
                    source_end=source_start + take,
                    prefix_length=prefixes[request_idx],
                    processed_length=done,
                    new_length=take,
                    absolute_start=absolute_start,
                    absolute_end=absolute_end,
                    terminal=terminal,
                )
            )
            processed[request_idx] += take
            available -= take

        rounds.append(KimiK3ChunkRound(tuple(round_slices)))
    return tuple(rounds)


def validate_whole_chunk_prefill(
    inputs: PyModelInputs,
    query_budget_tokens: int,
    *,
    tp_size: int,
    ep_size: int,
    alignment_tokens: int,
) -> None:
    """Reject unsupported whole-chunk modes before any cache mutation."""

    attention_inputs = inputs.attention_inputs
    if attention_inputs is None:
        raise RuntimeError("whole-model K3 Prefill requires attention inputs")
    if alignment_tokens % 64:
        raise RuntimeError(
            "whole-model K3 Prefill requires a checkpoint alignment "
            "divisible by the cuLA checkpoint step 64; "
            f"alignment_tokens={alignment_tokens}"
        )
    if query_budget_tokens % tp_size:
        raise RuntimeError(
            "KIMI_K3_PREFILL_CHUNK_TOKENS must be divisible by attention TP; "
            f"chunk={query_budget_tokens}, TP={tp_size}"
        )
    if ep_size != tp_size:
        raise RuntimeError(
            "whole-model K3 Prefill requires TP == EP Sequence Parallel; "
            f"TP={tp_size}, EP={ep_size}"
        )
    if bool(getattr(attention_inputs, "is_target_verify", False)):
        raise RuntimeError("whole-model K3 Prefill does not support target verify")
    if bool(getattr(attention_inputs, "is_cuda_graph", False)):
        raise RuntimeError("whole-model K3 Prefill does not support CUDA Graph")
    if getattr(attention_inputs, "context_parallel_info", None) is not None:
        raise RuntimeError(
            "whole-model K3 Prefill does not support framework Prefill CP"
        )
    multimodal = inputs.multimodal_inputs
    features = multimodal.multimodal_features
    locs = multimodal.mm_features_locs_host
    loc_count = 0 if locs is None else int(locs.numel())
    if bool(features) != bool(loc_count):
        raise RuntimeError(
            "whole-model K3 Prefill requires matching multimodal features and "
            f"locations: features={len(features)} locations={loc_count}"
        )
    if len(features) != loc_count:
        raise RuntimeError(
            "whole-model K3 Prefill multimodal feature/location counts differ: "
            f"features={len(features)} locations={loc_count}"
        )
    if getattr(multimodal, "mm_extra_input", []):
        raise RuntimeError(
            "whole-model K3 Prefill does not support multimodal extra inputs"
        )


def host_lengths(value: torch.Tensor, name: str) -> list[int]:
    if value is None or not value.numel():
        raise RuntimeError(f"whole-model K3 Prefill requires {name}")
    source = value if value.device.type == "cpu" else value.detach().cpu()
    return [int(item) for item in source.tolist()]


def _select_batch_rows(
    value: torch.Tensor,
    host_indices: torch.Tensor,
    device_indices: torch.Tensor,
    *,
    batch_dim: int = 0,
) -> torch.Tensor:
    if value is None or not value.numel():
        return value
    if batch_dim < 0 or batch_dim >= value.ndim:
        raise RuntimeError(
            "whole-model K3 block-table batch dimension is invalid: "
            f"shape={tuple(value.shape)} batch_dim={batch_dim}"
        )
    indices = host_indices.tolist()
    if min(indices) < 0 or max(indices) >= int(value.shape[batch_dim]):
        raise RuntimeError(
            "whole-model K3 active row is outside block table: "
            f"shape={tuple(value.shape)} batch_dim={batch_dim} indices={indices}"
        )
    index = device_indices if value.device.type == "cuda" else host_indices
    return value.index_select(batch_dim, index).contiguous()


def _select_group_batch_rows(
    values: Sequence[torch.Tensor],
    host_indices: torch.Tensor,
    device_indices: torch.Tensor,
    *,
    padding_rows: int = 0,
) -> list[torch.Tensor]:
    return [
        _append_batch_rows(
            _select_batch_rows(value, host_indices, device_indices),
            padding_rows,
        )
        for value in values
    ]


def _append_batch_rows(
    value: Optional[torch.Tensor],
    rows: int,
    *,
    batch_dim: int = 0,
    fill_value: int = 0,
) -> Optional[torch.Tensor]:
    if value is None or not value.numel() or rows == 0:
        return value
    if rows < 0 or batch_dim < 0 or batch_dim >= value.ndim:
        raise ValueError(
            "whole-model K3 padding requires non-negative rows and a valid "
            f"batch dimension: rows={rows} shape={tuple(value.shape)} "
            f"batch_dim={batch_dim}"
        )
    padding_shape = list(value.shape)
    padding_shape[batch_dim] = rows
    padding = value.new_full(padding_shape, fill_value)
    return torch.cat((value, padding), dim=batch_dim)


def _block_table_batch_dim(value: Optional[torch.Tensor]) -> int:
    return 1 if value is not None and value.ndim == 3 else 0


def _round_padding_tokens(logical_tokens: int, tp_size: int) -> int:
    if logical_tokens <= 0:
        raise ValueError(
            "whole-model K3 chunk round requires at least one logical token"
        )
    if tp_size <= 0:
        raise ValueError(f"attention TP size must be positive, got {tp_size}")
    return (-logical_tokens) % tp_size


def _host_and_device_tensor(
    values: Sequence[int], dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build metadata in pinned host memory and enqueue an ordered H2D copy.

    Consumers run on the current CUDA stream, so non_blocking=True preserves
    producer/copy/consumer ordering without synchronizing the CPU thread.
    """

    host = torch.tensor(
        values,
        dtype=dtype,
        device="cpu",
        pin_memory=device.type == "cuda",
    )
    if device.type == "cpu":
        return host, host
    return host, host.to(device=device, non_blocking=True)


def _slice_token_aligned_tensor(
    value: Optional[torch.Tensor],
    input_ids: torch.Tensor,
    round_plan: KimiK3ChunkRound,
    name: str,
    *,
    padding_tokens: int = 0,
    padding_value: int = 0,
) -> Optional[torch.Tensor]:
    if value is None or not value.numel():
        return value
    if value.ndim == 0 or value.shape[0] != input_ids.numel():
        raise RuntimeError(
            f"whole-model K3 Prefill requires token-aligned {name}: "
            f"shape={tuple(value.shape)} tokens={input_ids.numel()}"
        )
    sliced = torch.cat(
        [
            value.narrow(0, item.source_start, item.new_length)
            for item in round_plan.slices
        ],
        dim=0,
    )
    return _append_batch_rows(
        sliced,
        padding_tokens,
        fill_value=padding_value,
    )


def _build_chunk_multimodal_inputs(
    multimodal_inputs: Optional[PyMultimodalInputs],
    round_plan: KimiK3ChunkRound,
    input_lengths: Sequence[int],
    *,
    device: torch.device,
) -> PyMultimodalInputs:
    chunk = PyMultimodalInputs()
    if multimodal_inputs is None or not multimodal_inputs.multimodal_features:
        return chunk

    features = multimodal_inputs.multimodal_features
    locs = multimodal_inputs.mm_features_locs_host
    if locs is None or locs.numel() != len(features):
        loc_count = 0 if locs is None else int(locs.numel())
        raise RuntimeError(
            "whole-model K3 Prefill requires one location per multimodal "
            f"feature: features={len(features)} locations={loc_count}"
        )
    if getattr(multimodal_inputs, "mm_extra_input", []):
        raise RuntimeError(
            "whole-model K3 Prefill does not support multimodal extra inputs"
        )

    request_offsets = [0]
    for length in input_lengths:
        if int(length) <= 0:
            raise RuntimeError(
                "whole-model K3 Prefill requires positive request lengths"
            )
        request_offsets.append(request_offsets[-1] + int(length))

    source_locs = [int(value) for value in locs.cpu().view(-1).tolist()]
    feature_owners: list[int] = []
    for feature, feature_start in zip(features, source_locs):
        if feature.ndim != 2 or feature.shape[0] <= 0:
            raise RuntimeError(
                "whole-model K3 Prefill requires non-empty 2-D multimodal "
                f"features, got shape={tuple(feature.shape)}"
            )
        # A partially reused image can start before its request's packed-token
        # range. Its final row still lies in the owning request, which makes the
        # ownership unambiguous and prevents those prefix rows from leaking into
        # the previous request's chunk.
        feature_last = feature_start + int(feature.shape[0]) - 1
        owner = next(
            (
                index
                for index, (start, end) in enumerate(
                    zip(request_offsets, request_offsets[1:])
                )
                if start <= feature_last < end
            ),
            None,
        )
        if owner is None:
            raise RuntimeError(
                "whole-model K3 Prefill multimodal feature is outside the "
                f"packed requests: loc={feature_start} rows={feature.shape[0]}"
            )
        feature_owners.append(owner)

    chunk_features: list[torch.Tensor] = []
    chunk_locs: list[int] = []
    packed_offset = 0
    for item in round_plan.slices:
        slice_start = int(item.source_start)
        slice_end = int(item.source_end)
        for feature, feature_start, owner in zip(features, source_locs, feature_owners):
            if owner != int(item.original_batch_idx):
                continue
            feature_end = feature_start + int(feature.shape[0])
            intersection_start = max(slice_start, feature_start)
            intersection_end = min(slice_end, feature_end)
            if intersection_start >= intersection_end:
                continue
            feature_offset = intersection_start - feature_start
            feature_length = intersection_end - intersection_start
            chunk_features.append(feature.narrow(0, feature_offset, feature_length))
            chunk_locs.append(packed_offset + intersection_start - slice_start)
        packed_offset += item.new_length

    if chunk_features:
        chunk.multimodal_features = chunk_features
        chunk.mm_features_locs_host, chunk.mm_features_locs = _host_and_device_tensor(
            chunk_locs, torch.int32, device
        )
    return chunk


def build_chunk_attention_inputs(
    attention_inputs: PyAttentionInputs,
    *,
    round_plan: KimiK3ChunkRound,
    device: torch.device,
    tp_size: int = 1,
) -> PyAttentionInputs:
    """Rebuild packed attention and block-table metadata for one round."""

    outer_physical_requests = int(attention_inputs.input_lengths_host.numel())
    logical_request_count = int(
        getattr(attention_inputs, "logical_request_count", 0)
        or outer_physical_requests
    )
    logical_round = logical_chunk_round(round_plan, logical_request_count)
    logical_tokens = int(logical_round.token_count)
    padding_tokens = _round_padding_tokens(logical_tokens, tp_size)
    padding_requests = int(padding_tokens > 0)
    lengths = [item.new_length for item in logical_round.slices]
    prefixes = [item.absolute_start for item in logical_round.slices]
    sequence_lengths = [item.absolute_end for item in logical_round.slices]
    batch_indices = [item.original_batch_idx for item in logical_round.slices]
    if padding_requests:
        # Every internal round is an independent model forward. Rebuild one
        # tail dummy request for that round instead of inheriting the outer
        # forward's dummy, whose length only aligns the whole packed batch.
        lengths.append(padding_tokens)
        prefixes.append(0)
        sequence_lengths.append(padding_tokens)
    total_tokens = logical_tokens + padding_tokens
    chunk = copy.copy(attention_inputs)
    cu_seqlens = [0]
    cu_kv_seqlens = [0]
    for length, sequence_length in zip(lengths, sequence_lengths):
        cu_seqlens.append(cu_seqlens[-1] + length)
        cu_kv_seqlens.append(cu_kv_seqlens[-1] + sequence_length)
    chunk.cu_seqlens_host, chunk.cu_seqlens = _host_and_device_tensor(
        cu_seqlens, torch.int32, device
    )
    _, chunk.cu_kv_seqlens = _host_and_device_tensor(cu_kv_seqlens, torch.int32, device)
    chunk.input_lengths_host, chunk.input_lengths = _host_and_device_tensor(
        lengths, torch.int32, device
    )
    chunk.prefix_lengths_host, chunk.prefix_lengths = _host_and_device_tensor(
        prefixes, torch.int32, device
    )
    chunk.sequence_lengths_host, chunk.sequence_lengths = _host_and_device_tensor(
        sequence_lengths, torch.int32, device
    )
    chunk.sequence_lengths_plus_1_d = chunk.sequence_lengths + 1
    max_length = max(lengths)
    # Only O(batch) Python metadata; output_size avoids a CUDA size readback.
    offsets = [i * max_length - cu_seqlens[i] for i in range(len(lengths))]
    chunk.padding_offset = torch.repeat_interleave(
        torch.tensor(offsets, dtype=torch.int32, device=device),
        chunk.input_lengths,
        output_size=total_tokens,
    )
    chunk.total_tokens = int(total_tokens)
    chunk.context_total_kv_length = int(sum(sequence_lengths))
    # Internal chunk rounds must not inherit the outer forward's layout. A
    # round can have a different TP remainder from the complete packed batch.
    chunk.logical_request_count = len(logical_round.slices)
    chunk.physical_request_count = len(logical_round.slices) + padding_requests
    chunk.logical_token_count = logical_tokens
    chunk.physical_token_count = int(total_tokens)
    chunk.is_s_padded = (
        chunk.logical_request_count != chunk.physical_request_count
        or chunk.logical_token_count != chunk.physical_token_count
    )
    chunk.is_prefill = True
    chunk.is_cuda_graph = False
    chunk.cache_store_inputs = None

    host_batch_indices, device_batch_indices = _host_and_device_tensor(
        batch_indices, torch.long, device
    )
    for name in (
        "kv_cache_block_id_host",
        "kv_cache_block_id_device",
        "kv_cache_kernel_block_id_host",
        "kv_cache_kernel_block_id_device",
    ):
        value = getattr(attention_inputs, name)
        batch_dim = _block_table_batch_dim(value)
        selected = _select_batch_rows(
            value,
            host_batch_indices,
            device_batch_indices,
            batch_dim=batch_dim,
        )
        selected = _append_batch_rows(
            selected,
            padding_requests,
            batch_dim=batch_dim,
        )
        if selected is not None:
            setattr(chunk, name, selected)
    chunk.kv_cache_block_id_host_by_group = _select_group_batch_rows(
        attention_inputs.kv_cache_block_id_host_by_group,
        host_batch_indices,
        device_batch_indices,
        padding_rows=padding_requests,
    )
    chunk.kv_cache_kernel_block_id_host_by_group = _select_group_batch_rows(
        attention_inputs.kv_cache_kernel_block_id_host_by_group,
        host_batch_indices,
        device_batch_indices,
        padding_rows=padding_requests,
    )
    chunk.kv_cache_kernel_block_id_device_by_group = _select_group_batch_rows(
        attention_inputs.kv_cache_kernel_block_id_device_by_group,
        host_batch_indices,
        device_batch_indices,
        padding_rows=padding_requests,
    )
    return chunk


def build_chunk_model_inputs(
    input_ids: torch.Tensor,
    attention_inputs: PyAttentionInputs,
    *,
    round_plan: KimiK3ChunkRound,
    multimodal_inputs: Optional[PyMultimodalInputs] = None,
    embedding_inputs: Optional[PyEmbeddingInputs] = None,
    force_disable_sp_run: bool = False,
    tp_size: int = 1,
) -> PyModelInputs:
    outer_physical_requests = int(attention_inputs.input_lengths_host.numel())
    logical_request_count = int(
        getattr(attention_inputs, "logical_request_count", 0)
        or outer_physical_requests
    )
    logical_round = logical_chunk_round(round_plan, logical_request_count)
    padding_tokens = _round_padding_tokens(logical_round.token_count, tp_size)
    chunk = PyModelInputs()
    logical_input_ids = torch.cat(
        [
            input_ids.narrow(0, item.source_start, item.new_length)
            for item in logical_round.slices
        ],
        dim=0,
    )
    chunk.input_ids = _append_batch_rows(logical_input_ids, padding_tokens)
    chunk.attention_inputs = build_chunk_attention_inputs(
        attention_inputs,
        round_plan=round_plan,
        device=input_ids.device,
        tp_size=tp_size,
    )
    chunk.multimodal_inputs = _build_chunk_multimodal_inputs(
        multimodal_inputs,
        logical_round,
        host_lengths(attention_inputs.input_lengths_host, "input_lengths_host"),
        device=input_ids.device,
    )
    if embedding_inputs is not None:
        chunk.embedding_inputs = PyEmbeddingInputs()
        combo_tokens_type_ids = _slice_token_aligned_tensor(
            getattr(embedding_inputs, "combo_tokens_type_ids", None),
            input_ids,
            logical_round,
            "combo_tokens_type_ids",
            padding_tokens=padding_tokens,
        )
        if combo_tokens_type_ids is not None:
            chunk.embedding_inputs.combo_tokens_type_ids = combo_tokens_type_ids
        text_tokens_mask = _slice_token_aligned_tensor(
            getattr(embedding_inputs, "text_tokens_mask", None),
            input_ids,
            logical_round,
            "text_tokens_mask",
            padding_tokens=padding_tokens,
            padding_value=1,
        )
        if text_tokens_mask is not None:
            chunk.embedding_inputs.text_tokens_mask = text_tokens_mask
    chunk.force_disable_sp_run = force_disable_sp_run
    return chunk


def prepare_round_fmha(fmha_impl: Any, attention_inputs: PyAttentionInputs) -> None:
    prepare = getattr(fmha_impl, "prepare", None)
    if not callable(prepare):
        raise RuntimeError(
            "whole-model K3 Prefill requires an FMHA implementation "
            "that can be replanned for each internal round"
        )
    prepare(attention_inputs)


def kda_materialized_block_maps(
    attention_inputs: PyAttentionInputs,
    *,
    layer_group_ids: Optional[Sequence[int]],
    kda_layer_indices: Sequence[int],
) -> Optional[tuple[torch.Tensor, ...]]:
    """Select active host block maps used to compact recurrent stores."""

    maps_by_group = getattr(
        attention_inputs, "kv_cache_kernel_block_id_host_by_group", None
    )
    if not maps_by_group or layer_group_ids is None:
        return None
    try:
        group_ids = sorted({int(layer_group_ids[index]) for index in kda_layer_indices})
    except IndexError as error:
        raise RuntimeError(
            "KDA layer/group map does not cover every KDA layer"
        ) from error
    if any(group_id < 0 or group_id >= len(maps_by_group) for group_id in group_ids):
        raise RuntimeError("KDA cache group is outside host kernel block maps")
    return tuple(maps_by_group[group_id] for group_id in group_ids)


def kda_round_state_mapping(
    round_plan: Optional[KimiK3ChunkRound],
    *,
    padding_original_batch_idx: Optional[int] = None,
) -> tuple[Optional[list[int]], Optional[list[bool]]]:
    if round_plan is None:
        return None, None
    active_indices = [item.original_batch_idx for item in round_plan.slices]
    continuation_mask = [item.processed_length > 0 for item in round_plan.slices]
    if padding_original_batch_idx is not None:
        active_indices.append(padding_original_batch_idx)
        continuation_mask.append(False)
    return active_indices, continuation_mask


__all__ = [
    "KimiK3ChunkCachePublisher",
    "KimiK3ChunkPublishContext",
    "KimiK3ChunkRound",
    "KimiK3ChunkRdmaPublisher",
    "KimiK3ChunkRdmaPublishStep",
    "KimiK3ChunkSlice",
    "build_chunk_model_inputs",
    "chunkwise_rdma_enabled",
    "host_lengths",
    "kda_materialized_block_maps",
    "kda_round_state_mapping",
    "logical_chunk_round",
    "plan_kimi_k3_chunk_rounds",
    "prepare_round_fmha",
    "validate_whole_chunk_prefill",
]


class KimiK3ChunkSession:
    def __init__(self, model):
        self.model = model
        self.plan = None
        self.publisher = None
        self.current_state_registry = None

    def run(self, inputs, fmha_impl=None, chunk_prefill_round_hook=None):
        model = self.model
        budget = model.execution_spec.chunk_tokens
        if (
            budget > 0
            and inputs.attention_inputs is not None
            and inputs.attention_inputs.is_prefill
            and inputs.input_ids.numel() > budget
        ):
            model._begin_whole_chunk_prefill(budget)
            try:
                return self._run_rounds(
                    inputs, fmha_impl, budget, chunk_prefill_round_hook
                )
            finally:
                model._end_whole_chunk_prefill()
                self.current_state_registry = None
        return model._forward_impl_one(inputs, fmha_impl)

    def _run_rounds(
        self,
        inputs: PyModelInputs,
        fmha_impl: Any,
        chunk_tokens: int,
        chunk_prefill_round_hook: Any = None,
    ) -> PyModelOutputs:
        model = self.model
        if model._kda_checkpoint_tokens is None:
            raise RuntimeError("Kimi K3 cache geometry is not initialized")
        validate_whole_chunk_prefill(
            inputs,
            chunk_tokens,
            tp_size=int(model.parallelism_config.get_attn_tp_size()),
            ep_size=int(model.parallelism_config.ep_size),
            alignment_tokens=model._kda_checkpoint_tokens,
        )
        input_ids = inputs.input_ids.reshape(-1)
        attention_inputs = inputs.attention_inputs
        total_tokens = int(input_ids.numel())
        input_lengths = host_lengths(
            attention_inputs.input_lengths_host, "input_lengths_host"
        )
        prefix_lengths = host_lengths(
            attention_inputs.prefix_lengths_host, "prefix_lengths_host"
        )
        logical_request_count = int(
            getattr(attention_inputs, "logical_request_count", 0)
            or len(input_lengths)
        )
        logical_input_lengths = input_lengths[:logical_request_count]
        logical_prefix_lengths = prefix_lengths[:logical_request_count]
        if sum(input_lengths) != total_tokens:
            raise RuntimeError(
                "whole-model K3 packed lengths do not cover input tokens: "
                f"lengths={sum(input_lengths)} tokens={total_tokens}"
            )
        if model._layer_group_ids is None:
            layer_map_host = getattr(
                attention_inputs, "kv_cache_layer_to_group_host", None
            )
            if layer_map_host is None or not layer_map_host.numel():
                raise RuntimeError(
                    "whole-model K3 Prefill requires a host layer/group map"
                )
            model._layer_group_ids = tuple(
                int(value) for value in layer_map_host.tolist()
            )
        # The outer C++ boundary may have appended a dummy request to align
        # the complete packed Prefill. Internal rounds have independent token
        # remainders, so plan real requests only and rebuild one round-local
        # dummy immediately before each model forward.
        rounds = plan_kimi_k3_chunk_rounds(
            logical_input_lengths,
            logical_prefix_lengths,
            chunk_budget=chunk_tokens,
            alignment_tokens=model._kda_checkpoint_tokens,
        )
        logical_rounds = tuple(logical_chunk_round(r, logical_request_count) for r in rounds)
        batch_plan = KimiK3BatchPlan.from_rounds(logical_rounds)
        self.plan = batch_plan
        chunk_cache_publisher = KimiK3ChunkCachePublisher.create(
            attention_inputs,
            model.kv_cache,
            model.layers,
            input_lengths=logical_input_lengths,
            prefix_lengths=logical_prefix_lengths,
            transfer_page_tokens=model._k3_page_tokens,
        )
        self.publisher = chunk_cache_publisher
        barrier(Group.TP)
        logging.info(
            "[K3_WHOLE_CHUNK_PREFILL] enabled total_tokens=%d "
            "requests=%d rounds=%d chunk_tokens=%d mla_page_tokens=%d "
            "checkpoint_tokens=%d shard_size=%d shard_rank=%d TP=%d EP=%d "
            "chunkwise_rdma=%s",
            total_tokens,
            len(input_lengths),
            len(rounds),
            chunk_tokens,
            model._k3_page_tokens,
            model._kda_checkpoint_tokens,
            model.kv_cache.local_shard_count,
            (
                int(model.parallelism_config.tp_rank)
                if model.parallelism_config.kv_page_rr_enabled()
                else 0
            ),
            int(model.parallelism_config.get_attn_tp_size()),
            int(model.parallelism_config.ep_size),
            chunk_cache_publisher.enabled,
        )
        terminal_hidden: Optional[torch.Tensor] = None
        terminal_mtp_hidden: Optional[torch.Tensor] = None
        force_disable_sp_run = inputs.force_disable_sp_run
        mtp_hidden_enabled = (
            chunk_prefill_round_hook is not None and not force_disable_sp_run
        )
        terminal_written = [False] * logical_request_count
        final_params: Any = None
        tp_size = int(model.parallelism_config.get_attn_tp_size())
        current_state_registry = KimiKDACurrentStateRegistry(
            logical_request_count + int(tp_size > 1)
        )
        self.current_state_registry = current_state_registry
        chunk_cache_publisher.publish_prefix()
        for round_idx, round_plan in enumerate(rounds):
            logical_round = logical_chunk_round(round_plan, logical_request_count)
            terminal_count = sum(int(item.terminal) for item in logical_round.slices)
            round_label = (
                f"round={round_idx},tokens={round_plan.token_count},"
                f"logical_tokens={logical_round.token_count},"
                f"requests={len(round_plan.slices)},"
                f"logical_requests={len(logical_round.slices)},"
                f"terminal={terminal_count}"
            )
            chunk_inputs = build_chunk_model_inputs(
                input_ids,
                attention_inputs,
                round_plan=round_plan,
                multimodal_inputs=inputs.multimodal_inputs,
                embedding_inputs=inputs.embedding_inputs,
                force_disable_sp_run=force_disable_sp_run,
                tp_size=tp_size,
            )
            selection = (
                batch_plan.draft_rows[round_idx].prepare(input_ids.device)
                if mtp_hidden_enabled else None
            )
            chunk_attention = chunk_inputs.attention_inputs
            round_label += (
                f",physical_tokens={chunk_attention.physical_token_count},"
                f"physical_requests={chunk_attention.physical_request_count}"
            )
            prepare_round_fmha(fmha_impl, chunk_attention)
            # The preceding round's draft forward has consumed the previous
            # target round. Drop that owner before this target forward builds
            # and all-gathers the next 3-layer Eagle hidden tensor.
            model._release_prefill_mtp_hidden_buffer()
            chunk_publish_context = chunk_cache_publisher.begin_round(logical_round)
            trace_id = 0
            if _trace_ops.forward_trace_active():
                trace_id = _trace_ops.record_forward_trace_chunk(
                    round_idx,
                    [item.original_batch_idx for item in logical_round.slices],
                    [item.new_length for item in logical_round.slices],
                    [item.absolute_start for item in logical_round.slices],
                    chunk_attention.physical_request_count,
                    chunk_attention.physical_token_count,
                )
            trace_scope = (
                torch.profiler.record_function(f"RTP::model_forward(id={trace_id})")
                if trace_id else nullcontext()
            )
            with trace_scope, torch.profiler.record_function(
                f"RTP::kimi_k3.chunk_prefill.target_forward({round_label})"
            ):
                round_output = model._forward_impl_one(
                    chunk_inputs,
                    fmha_impl,
                    kda_current_state_registry=current_state_registry,
                    round_plan=round_plan,
                    chunk_publish_context=chunk_publish_context,
                )
            if trace_id:
                _trace_ops.finish_forward_trace_chunk(trace_id)
            chunk_cache_publisher.commit_round(chunk_publish_context)
            if os.environ.get("KIMI_K3_SMOKE_EVIDENCE") == "1":
                logging.info(
                    "[K3_SMOKE_EVENT] %s",
                    json.dumps(
                        {
                            "kind": "chunk",
                            "rank": int(model.parallelism_config.get_attn_tp_rank()),
                            "round": round_idx,
                            "tp": tp_size,
                            "logical_tokens": logical_round.token_count,
                            "physical_tokens": chunk_attention.physical_token_count,
                            "logical_requests": len(logical_round.slices),
                            "physical_requests": chunk_attention.physical_request_count,
                        },
                        sort_keys=True,
                    ),
                )
            if chunk_prefill_round_hook is not None:
                # Target projections have consumed MLA's aliased output. Its
                # historical KV scratch must not overlap the draft's expansion.
                fmha_impl.release_forward_workspace()
                if mtp_hidden_enabled:
                    round_mtp_hidden = model.get_mtp_target_hidden_states(-1)
                    if round_mtp_hidden is None:
                        raise RuntimeError(
                            "whole-model K3 Prefill did not publish MTP target hidden rows"
                        )
                    if terminal_mtp_hidden is None:
                        terminal_mtp_hidden = torch.empty(
                            (logical_request_count, *round_mtp_hidden.shape[1:]),
                            dtype=round_mtp_hidden.dtype,
                            device=round_mtp_hidden.device,
                        )
                    assert selection is not None
                    for request_idx, row in batch_plan.terminal_rows[round_idx]:
                        terminal_mtp_hidden[request_idx].copy_(round_mtp_hidden[row])
                    model._mtp_hidden_buffer = selection.select(round_mtp_hidden)
                    model._mtp_hidden_valid_tokens = selection.token_count
                chunk_prefill_round_hook(logical_round, round_plan is rounds[-1])
            if terminal_hidden is None:
                terminal_hidden = torch.empty(
                    (logical_request_count, round_output.hidden_states.shape[-1]),
                    dtype=round_output.hidden_states.dtype,
                    device=round_output.hidden_states.device,
                )
            final_params = getattr(round_output, "params_ptr", None)
            packed_end = 0
            for item in logical_round.slices:
                packed_end += item.new_length
                if item.terminal:
                    terminal_hidden[item.original_batch_idx].copy_(
                        round_output.hidden_states[packed_end - 1]
                    )
                    terminal_written[item.original_batch_idx] = True
            del selection
            del chunk_inputs
            del chunk_attention
            del round_output
        chunk_cache_publisher.validate_complete()
        if not chunk_cache_publisher.enabled:
            model._publish_whole_chunk_cache(attention_inputs)
        if terminal_hidden is None or not all(terminal_written):
            missing = [
                idx for idx, written in enumerate(terminal_written) if not written
            ]
            raise RuntimeError(f"whole-model K3 missing terminal rows for {missing}")
        if mtp_hidden_enabled:
            if terminal_mtp_hidden is None:
                raise RuntimeError("whole-model K3 missing terminal MTP hidden rows")
            model._mtp_hidden_buffer = terminal_mtp_hidden
            model._mtp_hidden_valid_tokens = logical_request_count
        hidden = terminal_hidden
        result = (
            PyModelOutputs(hidden, final_params)
            if final_params is not None
            else PyModelOutputs(hidden)
        )
        result.lm_output_already_selected = True
        return result
