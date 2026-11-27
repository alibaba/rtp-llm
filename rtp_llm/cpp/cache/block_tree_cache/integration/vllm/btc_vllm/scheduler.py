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

"""Bind vLLM requests and device references to the standalone BTC core."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import count, islice
from typing import TYPE_CHECKING

from block_tree_cache import (
    Binding,
    Cache,
    Error,
    EventKind,
    Group,
    HostRegion,
    Limits,
    Operation,
    Page,
    PageState,
    Plan,
    Position,
    Receipt,
    Result,
    Rule,
    Status,
    Work,
    WorkKind,
)
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager

from btc_vllm.cache_integration import BtcCacheBlocks, BtcCacheIntegration
from btc_vllm.remote_executor import RemoteExecutor
from btc_vllm.torchtpu.rank_transfers import RankResult

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

    from btc_vllm.remote_executor import RemoteStorage
    from btc_vllm.connector import BtcHostHandshakeMetadata
    from btc_vllm.torchtpu.worker import BtcHostGeometry, BtcTpuHostRegion

_DEVICE_POOL = 0
_EXECUTOR = 0
_DEVICE_ROUTES = (WorkKind.HOST_TO_DEVICE, WorkKind.DEVICE_TO_HOST)
_REMOTE_ROUTES = (
    WorkKind.REMOTE_QUERY,
    WorkKind.REMOTE_TO_HOST,
    WorkKind.HOST_TO_REMOTE,
)


@dataclass(slots=True)
class _Admission:
    keys: tuple[bytes, ...]
    limits: Limits
    plan: Plan | None = None
    blocks: BtcCacheBlocks | None = None
    selected_slots: frozenset[int] = frozenset()


def step_device_slots(step: SchedulerOutput) -> set[int]:
    slots = set(step.new_block_ids_to_zero or ())
    for copy in step.kv_cache_block_copies or ():
        slots.update((copy.src_block_id, copy.dst_block_id))
    if connector_state := step.kv_connector_block_state:
        for handoffs in connector_state.boundary_state_offloads.values():
            slots.update(slot for _, slot, _ in handoffs)
    return slots


class BtcScheduler:
    def __init__(
        self,
        config: VllmConfig,
        geometry: BtcHostGeometry | None,
    ) -> None:
        transfer = config.kv_transfer_config
        assert transfer is not None
        extra = transfer.kv_connector_extra_config
        if "btc_v6d" in extra or "btc_tree_library" in extra:
            raise ValueError("Use btc_pace and the standalone block-tree-cache wheel")
        self.remote_config = extra.get("btc_pace")
        self._model_identity = (
            config.model_config.model,
            config.cache_config.cache_dtype,
        )
        self._geometry = geometry
        self._num_ranks = geometry.num_ranks if geometry is not None else 0
        self._capacity = geometry.num_slots if geometry is not None else 0
        self._remote: RemoteExecutor | None = None
        self._native: Cache | None = None
        self._local_host: BtcTpuHostRegion | None = None
        self._host_region: HostRegion | None = None
        self._holds: dict[int, Page] = {}
        self._load_keys: dict[int, tuple[bytes, ...]] = {}
        self._hold_ids = count(1)
        self._request_ids = count(1)
        self._requests: dict[str, int] = {}
        self._request_names: dict[int, str] = {}
        self._admissions: dict[str, _Admission] = {}
        self._deferred: set[int] = set()
        self._bypass: set[str] = set()
        self._device_work: dict[Operation, Work] = {}
        self._device_receipts: dict[Operation, dict[int, RankResult]] = {}
        self._finished: set[str] = set()
        self._failed: set[str] = set()

    @property
    def native(self) -> Cache:
        assert self._native is not None, "BTC worker layout has not been registered"
        return self._native

    def bind(self, manager: KVCacheManager) -> None:
        self.manager = manager
        self.pool = manager.block_pool
        self.step = self.pool.hash_block_size
        managers = manager.coordinator.single_type_managers
        self.group_sizes = tuple(group.block_size for group in managers)
        for group in managers:
            if group.block_pool is not self.pool or type(group) not in (
                FullAttentionManager,
                MambaManager,
            ):
                raise ValueError(
                    "BTC needs shared full-attention and Mamba device pages"
                )
            if isinstance(group, MambaManager) and group.mamba_cache_mode != "align":
                raise ValueError("BTC needs Mamba checkpoints aligned to whole pages")
        self.last_checkpoint = tuple(
            isinstance(group, MambaManager) for group in managers
        )
        self._groups = tuple(
            Group(
                group, Rule.LINEAR if checkpoint else Rule.FULL, size,
                member_pools=(_DEVICE_POOL,),
            )
            for group, (size, checkpoint) in enumerate(
                zip(self.group_sizes, self.last_checkpoint, strict=True)
            )
        )
        self.cache = BtcCacheIntegration(manager, self)
        if geometry := self._geometry:
            self._bind_local_host(geometry)

    def _bind_local_host(self, geometry: BtcHostGeometry) -> None:
        from btc_vllm.torchtpu.worker import open_tpu_host_region

        layout = (geometry.num_ranks, geometry.rank_bytes)
        self._local_host = open_tpu_host_region(geometry)
        try:
            memory = self._local_host.create_kv_memoryview().cast("B")
            self._host_region = HostRegion(
                memory, geometry.row_bytes, geometry.num_ranks * geometry.rank_bytes
            )
            if self.remote_config is None:
                self._open(layout)
            else:
                from btc_vllm.torchtpu.remote import BtcLocalPaceStorage

                config = dict(
                    self.remote_config, block_size=self.step, use_eagle_pop=False
                )
                assert self._local_host.fd is not None, "BTC Host pool needs a backing file"
                storage = BtcLocalPaceStorage(
                    geometry, config, self.last_checkpoint, memory=memory,
                    memory_fd=self._local_host.fd, resolve_slot=self._host_slot_buffer,
                )
                self.configure_remote(storage, layout)
        except BaseException:
            self._close_local_host()
            raise

    def _host_slot_buffer(self, group: int, slot: int) -> memoryview:
        # Reset replaces the cache handle, while the SDK keeps the same mapping.
        return self.native.host_slot_buffer(group, slot)

    def finish_host_attachment(
        self, metadata: Mapping[tuple[int, int], BtcHostHandshakeMetadata]
    ) -> None:
        from btc_vllm.connector import BtcHostHandshakeMetadata

        assert self._local_host is not None, "BTC scheduler Host storage is unbound"
        backing = self._local_host.backing_identity()
        reports = metadata.values()
        if (
            len(reports) != self._num_ranks
            or any(not isinstance(item, BtcHostHandshakeMetadata) or item.backing != backing
                   for item in reports)
            or {item.rank for item in reports} != set(range(self._num_ranks))
        ):
            raise ValueError("BTC TPU workers did not all attach to the scheduler's Host backing")
        if not self._local_host.detach_name():
            raise RuntimeError("BTC Host rendezvous no longer names the attached backing")

    def _close_local_host(self) -> None:
        if self._host_region is not None:
            memory = self._host_region.buffer
            assert isinstance(memory, memoryview)
            memory.release()
            self._host_region = None
        if self._local_host is not None:
            self._local_host.cleanup()
            self._local_host = None

    def configure_host(self, num_ranks: int, capacity: int) -> None:
        assert self._native is None, "BTC Host layout was already registered"
        self._num_ranks = num_ranks
        self._capacity = capacity

    def configure_remote(self, storage: RemoteStorage, layout: object) -> None:
        try:
            assert self.remote_config is not None and storage.capacity == self._capacity
            self._remote = RemoteExecutor(storage, self.remote_config)
            self._open(layout)
        except BaseException:
            if self._remote is None:
                storage.close()
            else:
                self._remote.close()
                self._remote = None
            raise

    def _open(self, layout: object) -> None:
        self._layout_identity = layout
        namespace = json.dumps(
            (1, self._model_identity, self.group_sizes, self.last_checkpoint, layout),
            separators=(",", ":"),
        ).encode()
        routes = {kind: list(range(self._num_ranks)) for kind in _DEVICE_ROUTES}
        if self._remote is not None:
            routes.update({kind: [_EXECUTOR] for kind in _REMOTE_ROUTES})
        self._native = Cache(
            namespace, self.step, self._groups, host_slots=self._capacity, routes=routes,
            host_region=self._host_region, tiered_eviction=True,
        )

    def _hold(self, slot: int) -> Page:
        identity = next(self._hold_ids)
        page = Page(_DEVICE_POOL, slot, identity, identity)
        assert slot not in self._holds, f"BTC already owns device slot {slot}"
        self.pool.touch([self.pool.blocks[slot]])
        self._holds[slot] = page
        return page

    def _release(self, page: Page, *, evict_hash: bool = False) -> None:
        assert self._holds.pop(page.slot) == page, "BTC released a stale device hold"
        self._load_keys.pop(page.hold, None)
        block = self.pool.blocks[page.slot]
        if evict_hash and block.ref_cnt == 1:
            self.pool.evict_blocks({page.slot})
        self.pool.free_blocks([block])

    def publish(self, request: Request, committed: int) -> None:
        hashes = tuple(request.block_hashes[: committed // self.step])
        rows = self.manager.get_blocks(request.request_id).blocks
        bindings = []
        for group, row in enumerate(rows):
            size = self.group_sizes[group]
            for index, block in enumerate(row[: committed // size], start=1):
                end = index * size
                if block.is_null or block.block_hash_num_tokens != end:
                    continue
                if block.block_id in self._holds or end // self.step > len(hashes):
                    continue
                key = hashes[end // self.step - 1]
                if block.block_hash != make_block_hash_with_group_id(key, group):
                    continue
                bindings.append(
                    Binding(Position(group, end), (self._hold(block.block_id),))
                )
        if not bindings:
            return
        try:
            status = self.native.publish(hashes, committed, bindings)
        except BaseException:
            for binding in bindings:
                self._release(binding.pages[0])
            raise
        if status is Status.RETRY:
            for binding in bindings:
                self._release(binding.pages[0])
        else:
            assert status is Status.OK, f"Unexpected BTC publication status {status}"

    def invalidate(self, slot: int) -> None:
        page = self._holds.get(slot)
        if page is not None:
            self.native.invalidate([page])
            self._advance()

    def reclaim(self, required_free: int, protected: set[int]) -> None:
        shortage = required_free - self.pool.get_num_free_blocks()
        if shortage <= 0:
            return
        self.native.reclaim(_DEVICE_POOL, shortage, self._eligible_device_pages(protected))
        self._advance()

    def _eligible_device_pages(self, protected: set[int]) -> list[Page]:
        protected = protected | self._selected_device_slots()
        return [page for page in self._holds.values() if self._device_page_is_idle(page, protected)]

    def _selected_device_slots(self) -> set[int]:
        return {slot for admission in self._admissions.values() for slot in admission.selected_slots}

    def _device_page_is_idle(self, page: Page, protected: set[int]) -> bool:
        slot = page.slot
        return (self._holds.get(slot) == page and slot not in protected
                and slot not in self.pool._reuse_watchers and self.pool.blocks[slot].ref_cnt == 1)

    def balance_device(self, protected: set[int]) -> None:
        total = self.pool.num_gpu_blocks - 1  # The null block never joins the free queue.
        free = self.pool.get_num_free_blocks()
        if total and self.native.device_balance_needed(_DEVICE_POOL, total, free):
            self.native.balance_device(
                _DEVICE_POOL, total, free, self._eligible_device_pages(protected),
            )

    def match(self, request: Request, maximum: int | None) -> tuple[KVCacheBlocks, int]:
        rid = request.request_id
        empty = self.manager.empty_kv_cache_blocks
        if not self.manager.prefix_cache_lookup_enabled(request) or rid in self._bypass:
            self.abandon_admission(rid)
            return empty, 0
        hashes = request.block_hashes
        limit = min(request.num_tokens - 1, len(hashes) * self.step)
        if maximum is not None:
            limit = min(limit, maximum)
        limit = max(0, limit)
        cap = (request.kv_transfer_params or {}).get("max_load_tokens")
        if cap is not None and (type(cap) is not int or cap < 0):
            self.abandon_admission(rid)
            raise ValueError("max_load_tokens must be a nonnegative integer")
        limits = Limits(
            limit, allow_remote=self._remote is not None and cap != 0,
            max_load_tokens=None if cap is None else min(cap, limit),
        )
        keys = tuple(islice(hashes, limit // self.step))
        admission = self._admissions.get(rid)
        if admission is not None and (keys != admission.keys or limits != admission.limits):
            self.abandon_admission(rid)
            admission = None
        try:
            if admission is None:
                self._cancel_native_request(rid)
                request_id = next(self._request_ids)
                self._requests[rid] = request_id
                self._request_names[request_id] = rid
                admission = self._admissions[rid] = _Admission(keys, limits)
                admission.plan = self.native.match(request_id, keys, limits)
            elif admission.plan is None:
                admission.plan = self.native.poll_match(self._requests[rid])
        except Error as error:
            self.abandon_admission(rid)
            if error.status is Status.RETRY:
                return empty, 0
            raise
        except BaseException:
            self.abandon_admission(rid)
            raise
        plan = admission.plan
        if plan is None:
            return empty, 0
        if admission.blocks is None:
            try:
                admission.blocks = self._project_admission(request, plan)
                admission.selected_slots = frozenset(
                    block.block_id for row in admission.blocks.blocks for block in row if not block.is_null
                )
            except BaseException:
                self.abandon_admission(rid)
                raise
        return admission.blocks, plan.ready_tokens

    def _project_admission(self, request: Request, plan: Plan) -> BtcCacheBlocks:
        device = plan.ready_tokens
        if (
            device and self.pool.enable_kv_cache_events
            and request.sampling_params is not None
            and request.kv_cache_report_mode == "full"
        ):
            for group, size in enumerate(self.group_sizes):
                end = device // size
                start = end - 1 if self.last_checkpoint[group] else 0
                self.cache.emit_cached_block_events(
                    request, end - start, size, group, first_block=start
                )
        rows = [
            [self.pool.null_block] * (plan.recoverable_tokens // size)
            for size in self.group_sizes
        ]
        loads = [[] for _ in rows]
        joins = [[] for _ in rows]
        for entry in plan.pages:
            group = entry.position.group
            pos = self._row(entry.position)
            if entry.state == PageState.LOAD:
                loads[group].append(pos)
                continue
            (page,) = entry.pages
            assert self._holds.get(page.slot) == page, "BTC matched a stale device hold"
            rows[group][pos] = self.pool.blocks[page.slot]
            if entry.state == PageState.JOIN:
                joins[group].append(pos)
        return BtcCacheBlocks(
            tuple(rows),
            tuple(map(tuple, loads)) if any(loads) else None,
            tuple(map(tuple, joins)) if any(joins) else None,
            plan,
        )

    def _row(self, position: Position) -> int:
        return position.end_token // self.group_sizes[position.group] - 1

    def external(self, request: Request) -> tuple[int | None, bool]:
        admission = self._admissions.get(request.request_id)
        if admission is None:
            return 0, False
        plan = admission.plan
        if plan is None:
            return None, False
        count = plan.recoverable_tokens - plan.ready_tokens
        return count, count > 0

    def bind_allocated(self, request: Request, plan: Plan) -> bool:
        rid = request.request_id
        admission = self._admissions[rid]
        assert admission.plan is plan, "BTC cannot bind a plan from another admission"
        rows = self.manager.get_blocks(rid).blocks
        bindings = []
        try:
            for entry in plan.pages:
                if entry.state != PageState.LOAD:
                    continue
                position = entry.position
                block = rows[position.group][self._row(position)]
                bindings.append(Binding(position, (self._hold(block.block_id),)))
            status = self.native.bind(plan, bindings)
        except BaseException:
            for binding in bindings:
                self._release(binding.pages[0])
            raise
        if status is Status.RETRY:
            for binding in bindings:
                self._release(binding.pages[0])
            return False
        assert status in (Status.OK, Status.WAITING), (
            f"Unexpected BTC bind status {status}"
        )
        if bindings:
            for binding in bindings:
                self._load_keys[binding.pages[0].hold] = admission.keys
        if status is Status.WAITING:
            self._deferred.add(self._requests[rid])
        del self._admissions[rid]
        plan.close()
        return True

    def allocated(self, request: Request, blocks: KVCacheBlocks, tokens: int) -> None:
        request_id = self._requests.get(request.request_id)
        if request_id is not None:
            self._deferred.discard(request_id)

    def _complete_release(self, work: Work, *, evict_hash: bool = False) -> None:
        for item in work.pages:
            for page in item.source:
                self._release(page, evict_hash=evict_hash)
        self.native.complete([Receipt(work.id, _EXECUTOR, Result.SUCCESS, True)])

    def _advance(self) -> tuple[Work, ...]:
        if self._remote is not None:
            receipts = self._remote.completed()
            if receipts:
                self.native.complete(receipts)
        batch = self.native.poll()
        for work in batch.work:
            if work.kind is WorkKind.RELEASE:
                self._complete_release(work, evict_hash=True)
            elif work.kind in _REMOTE_ROUTES:
                assert self._remote is not None, "BTC backend did not register"
                if self._remote.submit(work):
                    self.native.accept(work.id, _EXECUTOR)
        for event in batch.events:
            rid = self._request_names.get(event.request)
            if rid is None:
                continue
            if event.kind is EventKind.READY:
                self._finished.add(rid)
            elif event.kind is EventKind.RECOMPUTE:
                self._bypass.add(rid)
                self._failed.add(rid)
        if batch.events:
            self.native.ack_events(batch.events[-1].cursor)
        return batch.work

    def dispatch(self, step: SchedulerOutput) -> dict[Operation, Work]:
        protected = step_device_slots(step)
        batch = {}
        self.balance_device(protected)
        protected.update(self._selected_device_slots())
        rejected = []
        for work in self._advance():
            if (
                work.kind not in _DEVICE_ROUTES
                or work.request in self._deferred
                or work.id in self._device_work
            ):
                continue
            if work.kind is WorkKind.DEVICE_TO_HOST and any(
                not self._device_page_is_idle(page, protected)
                for item in work.pages for page in item.source
            ):
                rejected.extend(Receipt(work.id, rank, Result.REJECTED, True)
                                for rank in work.participants)
                continue
            for participant in work.participants:
                self.native.accept(work.id, participant)
            self._device_work[work.id] = work
            self._device_receipts[work.id] = {}
            batch[work.id] = work
        if rejected:
            self.native.complete(rejected)
        return batch

    def settle(
        self, receipts: dict[Operation, dict[int, RankResult]], protected: set[int],
    ) -> tuple[set[str], set[str]]:
        completed = []
        settled = []
        if any((work := self._device_work.get(op)) is not None
               and work.kind is WorkKind.DEVICE_TO_HOST for op in receipts):
            # Refresh eligibility before native uses real rank receipts to commit the move.
            self.balance_device(protected)
        for operation, ranks in receipts.items():
            work = self._device_work.get(operation)
            if work is None:
                continue
            seen = self._device_receipts[operation]
            for rank, result in ranks.items():
                assert rank in work.participants, f"Unexpected BTC participant {operation}/{rank}"
                previous = seen.get(rank)
                if previous is not None and previous.safe:
                    continue
                seen[rank] = RankResult(
                    ok=result.ok and (previous is None or previous.ok), safe=result.safe,
                )
                completed.append(
                    Receipt(
                        operation, rank,
                        Result.SUCCESS if result.ok else Result.FAILED, result.safe,
                    )
                )
            if len(work.participants) == len(seen) and all(result.safe for result in seen.values()):
                settled.append((work, seen))
                del self._device_work[operation], self._device_receipts[operation]
        if completed:
            self.native.complete(completed)
        self._advance()
        for work, ranks in settled:
            if work.kind is WorkKind.HOST_TO_DEVICE:
                if all(result.ok for result in ranks.values()):
                    self._publish_loaded(work)
                for item in work.pages:
                    for page in item.target:
                        self._load_keys.pop(page.hold, None)
        finished, failed = self._finished, self._failed
        self._finished, self._failed = set(), set()
        return finished, failed

    def _publish_loaded(self, work: Work) -> None:
        pending: dict[int, tuple[tuple[bytes, ...], list[Binding]]] = {}
        for item in work.pages:
            (page,) = item.target
            if self._holds.get(page.slot) != page:
                continue
            keys = self._load_keys[page.hold]
            pending.setdefault(id(keys), (keys, []))[1].append(Binding(item.position, item.target))
        for keys, bindings in pending.values():
            installed = self.native.installed_device_pages(keys, bindings)
            for binding, owns in zip(bindings, installed, strict=True):
                if owns:
                    position = binding.position
                    self.cache.register_loaded_block(
                        make_block_hash_with_group_id(BlockHash(keys[position.end_token // self.step - 1]),
                                                      position.group),
                        self.pool.blocks[binding.pages[0].slot], position.end_token,
                    )

    def _cancel_native_request(self, rid: str) -> None:
        key = self._requests.pop(rid, None)
        if key is not None:
            del self._request_names[key]
            self._deferred.discard(key)
            self.native.cancel(key)

    def abandon_admission(self, rid: str, plan: Plan | None = None) -> None:
        admission = self._admissions.get(rid)
        if admission is None or (plan is not None and admission.plan is not plan):
            return
        del self._admissions[rid]
        try:
            self._cancel_native_request(rid)
        finally:
            if admission.plan is not None:
                admission.plan.close()

    def finish_request(self, rid: str) -> None:
        self.abandon_admission(rid)
        self._cancel_native_request(rid)
        self._bypass.discard(rid)
        self._finished.discard(rid)
        self._failed.discard(rid)

    def prepare_reset(self) -> None:
        for rid in tuple(self._admissions):
            self.abandon_admission(rid)
        used = self.pool.num_gpu_blocks - self.pool.get_num_free_blocks()
        if self.busy() or used != len(self._holds) + 1:
            return
        if all(self.pool.blocks[slot].ref_cnt == 1 for slot in self._holds):
            self.native.invalidate(tuple(self._holds.values()))
            self._advance()

    def reset(self) -> bool:
        if self._holds or self.busy():
            return False
        for rid in tuple(self._admissions):
            self.abandon_admission(rid)
        self.native.drain()
        if self.native.close() is not Status.OK:
            return False
        self._native = None
        self._open(self._layout_identity)
        return True

    def busy(self) -> bool:
        return self._native is not None and bool(
            self.native.pending_work() or self._finished or self._failed
        )

    def awaiting_spill(self) -> bool:
        return self.busy() and any(
            self.pool.blocks[slot].ref_cnt == 1 for slot in self._holds
        )

    def close(self) -> None:
        if self._native is None:
            # Startup or reset may have left a region without a cache handle.
            if self._remote is not None:
                self._remote.close()
                self._remote = None
            self._close_local_host()
            return
        for rid in tuple(self._admissions):
            self.abandon_admission(rid)
        self.native.drain()
        # The executor can stop workers by signal on failure. Accepted DMA
        # needs its own rank receipt even after executor shutdown returns.
        worker_receipts = [
            Receipt(work.id, rank, Result.REJECTED, True)
            for work in self.native.poll().work if work.kind in _DEVICE_ROUTES
            and work.id not in self._device_work
            for rank in work.participants
        ]
        self.native.complete(worker_receipts)
        if self._remote is not None:
            self.native.complete(self._remote.stop())
        for work in self.native.poll().work:
            if work.kind is WorkKind.RELEASE:
                self._complete_release(work)
            elif work.kind in _REMOTE_ROUTES and self._remote is not None:
                self.native.complete(self._remote.reject_unsubmitted(work))
        if self.native.close() is not Status.OK:
            raise RuntimeError("BTC shutdown is waiting for safe transfer receipts")
        self._native = None
        if self._remote is not None:
            self._remote.close()
            self._remote = None
        self._close_local_host()
