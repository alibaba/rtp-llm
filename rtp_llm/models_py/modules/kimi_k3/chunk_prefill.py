"""Whole-model chunk Prefill planning and round-input construction for Kimi K3."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Sequence

import torch

from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs

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
        return tuple(
            index for index, terminal in enumerate(self.terminal) if terminal
        )

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
        page_size: int,
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
        if page_size <= 0:
            raise ValueError(
                f"K3 chunk RDMA page size must be positive, got {page_size}"
            )
        self.page_size = int(page_size)
        self.kda_layer_indices = frozenset(
            int(value) for value in kda_layer_indices
        )
        self._frontier = [0] * len(self.input_lengths)
        self._terminal = [False] * len(self.input_lengths)
        self._published_kda: set[tuple[int, int]] = set()
        self._final_pages = tuple(
            (prefix + length + self.page_size - 1) // self.page_size
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
            [prefix // self.page_size for prefix in self.prefix_lengths],
            [False] * len(self.input_lengths),
        )

    def round_step(
        self, round_plan: KimiK3ChunkRound
    ) -> KimiK3ChunkRdmaPublishStep:
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
            expected_frontier = int(item.absolute_start) // self.page_size
            if self._frontier[index] != expected_frontier:
                raise RuntimeError(
                    "K3 chunk RDMA publication frontier does not match the round start: "
                    f"request={index} frontier={self._frontier[index]} "
                    f"expected={expected_frontier}"
                )
            absolute_end = int(item.absolute_end)
            if item.terminal:
                end = (absolute_end + self.page_size - 1) // self.page_size
            else:
                end = absolute_end // self.page_size
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
            missing = [
                index for index, value in enumerate(self._terminal) if not value
            ]
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
        page_size: int,
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
                page_size=page_size,
                kda_layer_indices=(
                    layer_idx
                    for layer_idx, layer in enumerate(layers)
                    if layer.is_kda
                ),
            ),
            layers=layers,
            kv_cache=kv_cache,
        )

    @property
    def enabled(self) -> bool:
        return self._publisher is not None

    def _context(
        self, step: KimiK3ChunkRdmaPublishStep
    ) -> KimiK3ChunkPublishContext:
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

    def commit_round(
        self, context: Optional[KimiK3ChunkPublishContext]
    ) -> None:
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
    page_size: int,
) -> tuple[KimiK3ChunkRound, ...]:
    """Split a packed Prefill batch at absolute MLA page boundaries."""

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
    if page_size <= 0:
        raise ValueError(f"K3 MLA page size must be positive, got {page_size}")
    if any(prefix < 0 for prefix in prefixes):
        raise ValueError(f"K3 prefix lengths must be non-negative, got {prefixes}")
    if any(prefix % page_size for prefix in prefixes):
        raise ValueError(
            "whole-model K3 chunk Prefill requires page-aligned prefixes: "
            f"prefixes={prefixes} page={page_size}"
        )
    if chunk_budget < page_size and any(length > chunk_budget for length in lengths):
        raise ValueError(
            "K3 chunk budget must cover one MLA page when a request spans "
            f"rounds: budget={chunk_budget}, page={page_size}"
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
                aligned_end = ((absolute_start + available) // page_size) * page_size
                take = aligned_end - absolute_start
                if take <= 0:
                    continue

            absolute_start = prefixes[request_idx] + done
            absolute_end = absolute_start + take
            if not terminal and absolute_end % page_size:
                raise AssertionError(
                    "non-terminal K3 chunk slice is not page aligned: "
                    f"request={request_idx} end={absolute_end} page={page_size}"
                )
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

        if not round_slices:
            pending = [
                idx
                for idx, (done, total) in enumerate(zip(processed, lengths))
                if done < total
            ]
            raise RuntimeError(
                "K3 chunk budget cannot advance any pending request to an "
                "MLA page boundary: "
                f"budget={chunk_budget}, page={page_size}, pending={pending}"
            )
        rounds.append(KimiK3ChunkRound(tuple(round_slices)))
    return tuple(rounds)


def validate_whole_chunk_prefill(
    inputs: PyModelInputs,
    chunk_tokens: int,
    *,
    tp_size: int,
    ep_size: int,
    page_size: Optional[int],
) -> None:
    """Reject unsupported whole-chunk modes before any cache mutation."""

    attention_inputs = inputs.attention_inputs
    if attention_inputs is None:
        raise RuntimeError("whole-model K3 Prefill requires attention inputs")
    if page_size is None:
        raise RuntimeError("whole-model K3 Prefill requires an initialized cache")
    if page_size <= 0 or page_size % 64:
        raise RuntimeError(
            "whole-model K3 Prefill requires a positive cache page size "
            "divisible by the cuLA checkpoint step 64; "
            f"page_size={page_size}"
        )
    if chunk_tokens <= 0 or chunk_tokens % tp_size:
        raise RuntimeError(
            "KIMI_K3_PREFILL_CHUNK_TOKENS must be divisible by attention TP; "
            f"chunk={chunk_tokens}, TP={tp_size}"
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
    if os.environ.get("SP_TYPE", "").lower() == "eagle3":
        raise RuntimeError("whole-model K3 Prefill does not support EAGLE3/MTP")
    multimodal = inputs.multimodal_inputs
    if multimodal.multimodal_features or (
        multimodal.mm_features_locs_host is not None
        and multimodal.mm_features_locs_host.numel()
    ):
        raise RuntimeError("whole-model K3 Prefill does not support multimodal input")


def host_lengths(value: torch.Tensor, name: str) -> list[int]:
    if value is None or not value.numel():
        raise RuntimeError(f"whole-model K3 Prefill requires {name}")
    source = value if value.device.type == "cpu" else value.detach().cpu()
    return [int(item) for item in source.tolist()]


def _select_batch_rows(
    value: torch.Tensor,
    indices: list[int],
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
    if min(indices) < 0 or max(indices) >= int(value.shape[batch_dim]):
        raise RuntimeError(
            "whole-model K3 active row is outside block table: "
            f"shape={tuple(value.shape)} batch_dim={batch_dim} indices={indices}"
        )
    index = torch.tensor(indices, dtype=torch.long, device=value.device)
    return value.index_select(batch_dim, index).contiguous()


def _select_group_batch_rows(
    values: Sequence[torch.Tensor], indices: list[int]
) -> list[torch.Tensor]:
    return [_select_batch_rows(value, indices) for value in values]


def build_chunk_attention_inputs(
    attention_inputs: PyAttentionInputs,
    *,
    round_plan: KimiK3ChunkRound,
    device: torch.device,
) -> PyAttentionInputs:
    """Rebuild packed attention and block-table metadata for one round."""

    lengths = [item.new_length for item in round_plan.slices]
    prefixes = [item.absolute_start for item in round_plan.slices]
    sequence_lengths = [item.absolute_end for item in round_plan.slices]
    batch_indices = [item.original_batch_idx for item in round_plan.slices]
    total_tokens = sum(lengths)
    chunk = copy.copy(attention_inputs)
    cu_seqlens = [0]
    cu_kv_seqlens = [0]
    for length, sequence_length in zip(lengths, sequence_lengths):
        cu_seqlens.append(cu_seqlens[-1] + length)
        cu_kv_seqlens.append(cu_kv_seqlens[-1] + sequence_length)
    chunk.cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    chunk.cu_kv_seqlens = torch.tensor(
        cu_kv_seqlens, dtype=torch.int32, device=device
    )
    chunk.input_lengths = torch.tensor(lengths, dtype=torch.int32, device=device)
    chunk.prefix_lengths = torch.tensor(prefixes, dtype=torch.int32, device=device)
    chunk.sequence_lengths = torch.tensor(
        sequence_lengths, dtype=torch.int32, device=device
    )
    chunk.sequence_lengths_plus_1_d = chunk.sequence_lengths + 1
    max_length = max(lengths)
    padding_offset: list[int] = []
    cumulative_padding = 0
    for length in lengths:
        padding_offset.extend([cumulative_padding] * length)
        cumulative_padding += max_length - length
    chunk.padding_offset = torch.tensor(
        padding_offset, dtype=torch.int32, device=device
    )
    chunk.cu_seqlens_host = torch.tensor(cu_seqlens, dtype=torch.int32)
    chunk.input_lengths_host = torch.tensor(lengths, dtype=torch.int32)
    chunk.prefix_lengths_host = torch.tensor(prefixes, dtype=torch.int32)
    chunk.sequence_lengths_host = torch.tensor(sequence_lengths, dtype=torch.int32)
    chunk.total_tokens = int(total_tokens)
    chunk.context_total_kv_length = int(sum(sequence_lengths))
    chunk.is_prefill = True
    chunk.is_cuda_graph = False
    chunk.cache_store_inputs = None

    for name in (
        "kv_cache_block_id_host",
        "kv_cache_kernel_block_id_host",
        "kv_cache_kernel_block_id_device",
    ):
        value = getattr(attention_inputs, name)
        batch_dim = (
            1
            if name == "kv_cache_block_id_host"
            and value is not None
            and value.ndim == 3
            else 0
        )
        selected = _select_batch_rows(value, batch_indices, batch_dim=batch_dim)
        if selected is not None:
            setattr(chunk, name, selected)
    chunk.kv_cache_block_id_host_by_group = _select_group_batch_rows(
        attention_inputs.kv_cache_block_id_host_by_group, batch_indices
    )
    chunk.kv_cache_kernel_block_id_host_by_group = _select_group_batch_rows(
        attention_inputs.kv_cache_kernel_block_id_host_by_group, batch_indices
    )
    chunk.kv_cache_kernel_block_id_device_by_group = _select_group_batch_rows(
        attention_inputs.kv_cache_kernel_block_id_device_by_group, batch_indices
    )
    return chunk


def build_chunk_model_inputs(
    input_ids: torch.Tensor,
    attention_inputs: PyAttentionInputs,
    *,
    round_plan: KimiK3ChunkRound,
) -> PyModelInputs:
    chunk = PyModelInputs()
    chunk.input_ids = torch.cat(
        [
            input_ids.narrow(0, item.source_start, item.new_length)
            for item in round_plan.slices
        ],
        dim=0,
    )
    chunk.attention_inputs = build_chunk_attention_inputs(
        attention_inputs,
        round_plan=round_plan,
        device=input_ids.device,
    )
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
        raise RuntimeError("KDA layer/group map does not cover every KDA layer") from error
    if any(group_id < 0 or group_id >= len(maps_by_group) for group_id in group_ids):
        raise RuntimeError("KDA cache group is outside host kernel block maps")
    return tuple(maps_by_group[group_id] for group_id in group_ids)


def kda_round_state_mapping(
    round_plan: Optional[KimiK3ChunkRound],
) -> tuple[Optional[list[int]], Optional[list[bool]]]:
    if round_plan is None:
        return None, None
    return (
        [item.original_batch_idx for item in round_plan.slices],
        [item.processed_length > 0 for item in round_plan.slices],
    )


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
    "plan_kimi_k3_chunk_rounds",
    "prepare_round_fmha",
    "validate_whole_chunk_prefill",
]
