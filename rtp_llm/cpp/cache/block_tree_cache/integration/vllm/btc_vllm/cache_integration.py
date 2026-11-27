# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Engine-local bindings for BTC pages in the vLLM cache allocator."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    BlockHashWithGroupId,
    KVCacheBlock,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
    resolve_block_hashes,
)
from vllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from block_tree_cache import Plan

    from btc_vllm.scheduler import BtcScheduler


@dataclass
class BtcCacheBlocks(KVCacheBlocks):
    load_positions: tuple[tuple[int, ...], ...] | None = None
    join_positions: tuple[tuple[int, ...], ...] | None = None
    plan: Plan | None = None


class _BtcBoundRow(list[KVCacheBlock]):
    """Keep pending BTC bindings with the block table that owns them."""

    __slots__ = ("adopted",)

    def __init__(self, blocks: list[KVCacheBlock], first_pending: int) -> None:
        super().__init__(blocks)
        self.adopted = {
            pos: blocks[pos].block_id
            for pos in range(first_pending, len(blocks))
            if not blocks[pos].is_null
        }


@dataclass
class _Allocation:
    request_id: str
    bindings: BtcCacheBlocks | None
    num_loads: int
    required_blocks: int | None = None
    full_sequence_gate: bool = False


class BtcCacheIntegration:
    """Extend the cache objects owned by one BTC scheduler."""

    def __init__(self, manager: KVCacheManager, btc: BtcScheduler) -> None:
        self.manager = manager
        self.btc = btc
        self.coordinator = manager.coordinator
        self.pool = manager.block_pool
        self._allocation: _Allocation | None = None
        self._pending_events: set[int] = set()

        self._allocate = manager.allocate_slots
        self._count = self.coordinator.get_num_blocks_to_allocate
        self._adopt = self.coordinator.allocate_new_computed_blocks
        self._cache = self.pool.cache_full_blocks
        self._remove = self.pool._remove_cached_block_hashes
        self._reset = self.pool.reset_prefix_cache

        # Only the objects created by this BTC engine get these hooks.
        vars(manager)["allocate_slots"] = self.allocate_slots
        vars(self.coordinator).update(
            get_num_blocks_to_allocate=self._count_slots,
            allocate_new_computed_blocks=self._adopt_slots,
        )
        vars(self.pool).update(
            cache_full_blocks=self._cache_full_blocks,
            _remove_cached_block_hashes=self._remove_hashes,
            reset_prefix_cache=self._reset_prefix_cache,
        )

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
        full_sequence_must_fit: bool = False,
        reserved_blocks: int = 0,
        has_scheduled_reqs: bool = True,
        skip_zeroing_group_ids: tuple[int, ...] = (),
    ) -> KVCacheBlocks | None:
        """Keep vLLM's allocation flow and reclaim unreferenced BTC pages."""
        plan = (
            new_computed_blocks.plan
            if isinstance(new_computed_blocks, BtcCacheBlocks)
            else None
        )
        if plan is None:
            # vLLM can discard a local hit before allocating a replacement row.
            self.btc.abandon_admission(request.request_id)
        bindings = (
            new_computed_blocks
            if isinstance(new_computed_blocks, BtcCacheBlocks)
            and (
                new_computed_blocks.load_positions or new_computed_blocks.join_positions
            )
            else None
        )
        num_loads = (
            sum(map(len, bindings.load_positions))
            if bindings is not None and bindings.load_positions is not None
            else 0
        )
        protected = (
            {
                block.block_id
                for row in new_computed_blocks.blocks
                for block in row
                if not block.is_null
            }
            if new_computed_blocks is not None
            else set()
        )
        allocate_options = dict(
            num_new_computed_tokens=num_new_computed_tokens,
            new_computed_blocks=new_computed_blocks,
            num_lookahead_tokens=num_lookahead_tokens,
            num_external_computed_tokens=num_external_computed_tokens,
            delay_cache_blocks=delay_cache_blocks,
            num_encoder_tokens=num_encoder_tokens,
            full_sequence_must_fit=full_sequence_must_fit,
            reserved_blocks=reserved_blocks,
            has_scheduled_reqs=has_scheduled_reqs,
            skip_zeroing_group_ids=skip_zeroing_group_ids,
        )
        assert self._allocation is None, "BTC allocation must run on one scheduler"
        attempt = _Allocation(request.request_id, bindings, num_loads)
        self._allocation = attempt
        result = None
        try:
            while True:
                attempt.required_blocks = None
                result = self._allocate(request, num_new_tokens, **allocate_options)
                required = attempt.required_blocks
                if result is not None:
                    break
                if required is None:
                    return None
                if (
                    not attempt.full_sequence_gate
                    and required == 0
                    and num_new_tokens == 0
                    and bindings is not None
                    and bindings.join_positions is not None
                ):
                    # The full-sequence gate has passed, and this step only
                    # adopts existing slots. Headroom cannot affect the join.
                    allocate_options.update(reserved_blocks=0, has_scheduled_reqs=False)
                    result = self._allocate(request, num_new_tokens, **allocate_options)
                    if result is None:
                        return None
                    break
                if has_scheduled_reqs and request.status in (
                    RequestStatus.WAITING,
                    RequestStatus.PREEMPTED,
                ):
                    required += self.manager.watermark_blocks
                if not attempt.full_sequence_gate:
                    required += reserved_blocks
                if required >= self.pool.num_gpu_blocks:
                    # Includes the Mamba same-step admission sentinel.
                    return None
                free_before = self.pool.get_num_free_blocks()
                self.btc.reclaim(required, protected)
                if self.pool.get_num_free_blocks() <= free_before:
                    return None
        finally:
            self._allocation = None
            if result is None and plan is not None:
                self.btc.abandon_admission(request.request_id, plan)
        if plan is None:
            return result
        bound = False
        try:
            assert request.num_computed_tokens == 0, "BTC binds only admission rows"
            bound = self.btc.bind_allocated(request, plan)
        finally:
            if not bound:
                try:
                    self.manager.free(request)
                finally:
                    self.btc.abandon_admission(request.request_id, plan)
        return result if bound else None

    def _count_slots(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
        prefill_end: int = 0,
    ) -> int:
        required = self._count(
            request_id,
            num_tokens,
            new_computed_blocks,
            num_encoder_tokens,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap,
            prefill_end,
        )
        attempt = self._allocation
        if attempt is not None:
            assert attempt.request_id == request_id
            required += attempt.num_loads
            attempt.required_blocks = required
            attempt.full_sequence_gate = apply_admission_cap
        return required

    def _adopt_slots(
        self,
        request_id: str,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
        skip_zeroing_group_ids: tuple[int, ...] = (),
    ) -> None:
        attempt = self._allocation
        bindings = attempt.bindings if attempt is not None else None
        if bindings is None:
            self._adopt(
                request_id,
                new_computed_blocks,
                num_local_computed_tokens,
                num_external_computed_tokens,
                skip_zeroing_group_ids,
            )
            return
        assert attempt is not None and attempt.request_id == request_id
        groups = self.coordinator.single_type_managers
        empty_positions = ((),) * len(groups)
        loads = (
            bindings.load_positions
            if bindings.load_positions is not None
            else empty_positions
        )
        joins = (
            bindings.join_positions
            if bindings.join_positions is not None
            else empty_positions
        )
        if len(new_computed_blocks) != len(groups) or not (
            len(loads) == len(joins) == len(groups)
        ):
            raise ValueError("BTC page bindings must have one row per cache group")

        end = num_local_computed_tokens + num_external_computed_tokens
        for group, manager in enumerate(groups):
            row = new_computed_blocks[group]
            pending = loads[group] + joins[group]
            skipped = max(0, manager.get_num_skipped_tokens(end)) // manager.block_size
            load_set = set(loads[group])
            if (
                end % manager.block_size
                or num_local_computed_tokens % manager.block_size
                or len(row) != end // manager.block_size
                or len(set(pending)) != len(pending)
                or any(pos < skipped or pos >= len(row) for pos in pending)
                or any(
                    block.is_null != (pos < skipped or pos in load_set)
                    for pos, block in enumerate(row)
                )
                or any(
                    not block.is_null
                    and (
                        not 0 <= block.block_id < self.pool.num_gpu_blocks
                        or self.pool.blocks[block.block_id] is not block
                    )
                    for block in row
                )
            ):
                raise ValueError(
                    f"Invalid BTC positional cache bindings for group {group}"
                )

        # The entire aligned prefix row is supplied. Let vLLM adopt every
        # group's existing blocks, then fill only the connector's LOAD holes.
        self._adopt(request_id, new_computed_blocks, end, 0)
        for group, manager in enumerate(groups):
            pending = loads[group] + joins[group]
            if not pending:
                continue
            row = manager.req_to_blocks[request_id]
            targets = self.pool.get_new_blocks(len(loads[group]))
            for position, target in zip(loads[group], targets):
                row[position] = target
            first = min(pending)
            manager.num_cached_block[request_id] = first
            manager.req_to_blocks[request_id] = _BtcBoundRow(row, first)

    def register_loaded_block(
        self,
        block_hash: BlockHashWithGroupId,
        block: KVCacheBlock,
        num_tokens: int,
    ) -> None:
        """Publish an HBM receipt and let the first waiter announce its event."""
        assert not block.is_null and block.ref_cnt > 0 and block.block_hash is None
        self.pool._insert_block_hash(block_hash, block, num_tokens)
        if self.pool.enable_kv_cache_events:
            self._pending_events.add(block.block_id)

    def _cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
        block_mask: list[bool] | None = None,
    ) -> None:
        if not isinstance(blocks, _BtcBoundRow) or not blocks.adopted:
            self._cache(
                request,
                blocks,
                num_cached_blocks,
                num_full_blocks,
                block_size,
                kv_cache_group_id,
                block_mask,
            )
            return
        hashes = resolve_block_hashes(
            request.block_hashes, self.pool.hash_block_size, block_size
        )
        for pos in range(num_cached_blocks, num_full_blocks):
            block = blocks[pos]
            if block.is_null or (
                block_mask is not None and not block_mask[pos - num_cached_blocks]
            ):
                continue
            if (
                blocks.adopted.pop(pos, None) != block.block_id
                or block.block_hash is None
            ):
                self._cache(
                    request, blocks, pos, pos + 1, block_size, kv_cache_group_id
                )
                continue
            assert (
                block.block_hash
                == make_block_hash_with_group_id(hashes[pos], kv_cache_group_id)
                and block.block_hash_num_tokens == (pos + 1) * block_size
            ), "Adopted BTC page has a different prefix or endpoint"
            if block.block_id in self._pending_events:
                self._pending_events.remove(block.block_id)
                self.emit_cached_block_events(
                    request, 1, block_size, kv_cache_group_id, first_block=pos
                )

    def emit_cached_block_events(
        self,
        request: Request,
        num_cached_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
        *,
        first_block: int,
    ) -> None:
        """Publish a contiguous run of cached pages at the actual token range."""
        if not self.pool.enable_kv_cache_events or not num_cached_blocks:
            return
        if first_block == 0:
            self.pool.emit_cached_block_events(
                request, num_cached_blocks, block_size, kv_cache_group_id
            )
            return
        hashes = resolve_block_hashes(
            request.block_hashes, self.pool.hash_block_size, block_size
        )
        stop = first_block + num_cached_blocks
        extra_keys_list = []
        mm_index = 0
        for pos in range(first_block, stop):
            extra_keys, mm_index = self.pool._generate_event_extra_keys(
                request, pos * block_size, (pos + 1) * block_size, mm_index
            )
            extra_keys_list.append(extra_keys)
        self.pool.kv_event_queue.append(
            self.pool._build_block_stored_event(
                request,
                block_hashes=[
                    maybe_convert_block_hash(hashes[pos])
                    for pos in range(first_block, stop)
                ],
                parent_block_hash=maybe_convert_block_hash(hashes[first_block - 1]),
                start_token_idx=first_block * block_size,
                end_token_idx=stop * block_size,
                block_size=block_size,
                kv_cache_group_id=kv_cache_group_id,
                extra_keys_list=extra_keys_list,
                lookahead_token_ids=(
                    [
                        self.pool._lookahead_token_ids(request, (pos + 1) * block_size)
                        for pos in range(first_block, stop)
                    ]
                    if self.pool.num_hash_lookahead_tokens
                    else None
                ),
            )
        )

    def _remove_hashes(self, block: KVCacheBlock) -> list[BlockHashWithGroupId]:
        self._pending_events.discard(block.block_id)
        if block.block_hash is not None:
            self.btc.invalidate(block.block_id)
        return self._remove(block)

    def _reset_prefix_cache(self) -> bool:
        self.btc.prepare_reset()
        if not self._reset():
            return False
        self._pending_events.clear()
        return True
