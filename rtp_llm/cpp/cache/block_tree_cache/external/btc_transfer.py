# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side BTC transfers over pinned device pages and Host slots."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import count
from math import lcm
from typing import TYPE_CHECKING

from btc_tree import HOST, DeviceResidency, Endpoint, JoinBinding, TreeCache

if TYPE_CHECKING:
    from btc_remote import BtcRemoteCache


_PageKey = tuple[int, bytes]
# Worker copy order: (group, device slot, Host slot).
_Copies = tuple[tuple[int, int, int], ...]


@dataclass(slots=True)
class DevicePage:
    group: int
    row: int
    slot: int
    generation: int
    block_hash: bytes

    @property
    def key(self) -> _PageKey:
        return self.group, self.block_hash

    @property
    def identity(self) -> DeviceResidency:
        return self.group, self.slot, self.generation


@dataclass(slots=True)
class _Job:
    load: bool
    copies: _Copies
    records: list[DevicePage]
    host_ticket: int
    receipts: dict[int, bool] = field(default_factory=dict)
    waiters: set[str] = field(default_factory=set)


class TransferScheduler:
    """Own transfer transactions and the device holds behind their tree entries.

    All methods run on the scheduler thread. The caller maps device slot IDs to
    physical pages and implements pin, unpin and publication callbacks. After
    dispatch, each job keeps its holds until every participant reports a result.
    """

    def __init__(
        self,
        tree: TreeCache,
        pin: Callable[[list[int]], None],
        unpin: Callable[[list[int]], None],
        publish_loaded: Callable[[DevicePage], None],
    ) -> None:
        self.tree = tree
        self._pin = pin
        self._unpin = unpin
        self._publish_loaded = publish_loaded
        self._ids = count(1)
        self._num_ranks = 0
        self._capacity = 0
        self._remote: BtcRemoteCache | None = None
        self.resident: dict[int, DevicePage] = {}
        self._reclaimable: set[int] = set()
        self._reclaiming: set[int] = set()
        self._by_key: dict[_PageKey, DevicePage] = {}
        self._incoming: dict[_PageKey, DevicePage] = {}
        # A publication batch shares one hash path until its Host offer is handled.
        self._offered: dict[int, tuple[bytes, ...]] = {}
        self._jobs: dict[int, _Job] = {}
        self._unsent: list[int] = []
        self._dependencies: dict[str, set[int]] = {}
        self._cancelled: set[str] = set()
        self._failed: set[str] = set()
        self._bypass: set[str] = set()
        self._closed = False

    def configure_host(self, num_ranks: int, capacity: int) -> None:
        assert self._capacity == 0, "BTC Host slots were already configured"
        if num_ranks <= 0:
            raise ValueError("BTC Host transfers require a positive rank count")
        self.tree.configure_host(capacity)
        self._num_ranks = num_ranks
        self._capacity = capacity

    def configure_layout(
        self, step: int, group_sizes: Sequence[int], last_checkpoint: Sequence[bool]
    ) -> None:
        if step <= 0 or any(size <= 0 or size % step for size in group_sizes):
            raise ValueError("BTC groups must align to the positive hash block size")
        if not group_sizes or len(group_sizes) != len(last_checkpoint):
            raise ValueError("BTC requires one checkpoint policy per cache group")
        self._step = step
        self._group_sizes = tuple(group_sizes)
        self._last_checkpoint = tuple(last_checkpoint)
        self._alignment = lcm(*group_sizes)

    def configure_remote(self, storage, config) -> None:
        from btc_remote import BtcRemoteCache

        assert self._remote is None and storage.capacity == self._capacity
        self._remote = BtcRemoteCache(self.tree, storage, config)

    def _remote_endpoints(self, limit: int) -> list[Endpoint]:
        endpoints = []
        for end in range(self._alignment, limit + 1, self._alignment):
            stop = end // self._step
            by_group = []
            for size, checkpoint in zip(self._group_sizes, self._last_checkpoint):
                ratio = size // self._step
                by_group.append(
                    range(stop - 1 if checkpoint else ratio - 1, stop, ratio)
                )
            endpoints.append((end, by_group))
        return endpoints

    def device_available(self, key: _PageKey) -> bool:
        return key in self._by_key or key in self._incoming

    def warm(self, request_id: str, hashes: Sequence[bytes], limit: int) -> bool:
        return self._remote is None or self._remote.warm(
            request_id, hashes, self._remote_endpoints(limit), self.device_available
        )

    def match(
        self, hashes: Sequence[bytes], limit: int
    ) -> tuple[int, int, list[JoinBinding], list[JoinBinding]]:
        return self.tree.match_whole_pages_with_join(
            hashes, limit, self._step, self._group_sizes, self._last_checkpoint
        )

    def validate_device_binding(
        self,
        group: int,
        block_hash: bytes,
        slot: int,
        generation: int,
        *,
        pending: bool,
    ) -> None:
        pages = self._incoming if pending else self._by_key
        page = pages[group, block_hash]
        assert page.slot == slot and page.generation == generation, (
            "BTC match returned a stale device binding"
        )

    def publish(
        self, hashes: tuple[bytes, ...], pages: Sequence[tuple[int, int, int]]
    ) -> None:
        """Publish committed pages and offer them for Host writeback."""
        records = []
        keys = set()
        for group, row, slot in pages:
            key = group, hashes[row]
            if self.device_available(key) or key in keys:
                continue
            assert slot not in self.resident, (
                "BTC device slot reused without invalidation"
            )
            records.append(DevicePage(group, row, slot, next(self._ids), key[1]))
            keys.add(key)
        if not records:
            return
        slots = [page.slot for page in records]
        self._pin(slots)
        try:
            self.tree.publish_device(
                hashes,
                (
                    (page.group, page.row, page.slot, page.generation)
                    for page in records
                ),
            )
        except BaseException:
            self._unpin(slots)
            raise
        for page in records:
            self.resident[page.slot] = self._by_key[page.key] = page
            self._offered[page.slot] = hashes

    def invalidate(self, slot: int) -> None:
        page = self.resident.get(slot)
        if page is None:
            return
        if slot not in self._reclaiming:
            self.tree.unpublish_device(*page.identity)
        del self.resident[slot]
        del self._by_key[page.key]
        self._offered.pop(slot, None)
        self._reclaimable.discard(slot)
        self._unpin([slot])

    def reclaim(
        self,
        shortage: int,
        eligible: Callable[[int], bool],
        evict: Callable[[int], bool],
    ) -> None:
        if shortage <= 0:
            return
        candidates = [
            page.identity
            for slot, page in self.resident.items()
            if slot in self._reclaimable and eligible(slot)
        ]
        if not candidates:
            return
        ticket, victims = self.tree.plan_device_reclaim(candidates, shortage)
        if not ticket:
            return
        released = []
        try:
            for identity in victims:
                _, slot, _ = identity
                self._reclaiming.add(slot)
                try:
                    if evict(slot):
                        released.append(identity)
                finally:
                    self._reclaiming.remove(slot)
        finally:
            self.tree.settle_device_reclaim(ticket, released)

    def begin_load(
        self,
        request_id: str,
        hashes: Sequence[bytes],
        loads: Sequence[tuple[int, int, int, int]],
        joins: set[int],
    ) -> None:
        """Start LOAD copies and register the request on every joined job."""
        assert request_id not in self._dependencies, (
            "BTC request already has a pending load"
        )
        dependencies = set(joins)
        if loads:
            records = [
                DevicePage(group, row, slot, next(self._ids), hashes[row])
                for group, row, slot, _ in loads
            ]
            job_id = next(self._ids)
            lease = self.tree.pin_host_load(
                (group, host) for group, _, _, host in loads
            )
            slots = [page.slot for page in records]
            try:
                self._pin(slots)
                try:
                    self.tree.begin_device_load(
                        hashes,
                        (
                            (
                                page.group,
                                page.row,
                                HOST,
                                host,
                                page.slot,
                                page.generation,
                            )
                            for page, (_, _, _, host) in zip(records, loads)
                        ),
                        job_id,
                    )
                except BaseException:
                    self._unpin(slots)
                    raise
            except BaseException:
                self.tree.release_host_load(lease)
                raise
            self._incoming.update((page.key, page) for page in records)
            copies = tuple((group, slot, host) for group, _, slot, host in loads)
            self._jobs[job_id] = _Job(True, copies, records, lease)
            self._unsent.append(job_id)
            dependencies.add(job_id)
        assert dependencies, "BTC external hit has no producer"
        self._dependencies[request_id] = dependencies
        for job_id in dependencies:
            self._jobs[job_id].waiters.add(request_id)

    def _schedule_store(self) -> None:
        if not self._offered:
            return
        candidates = [self.resident[slot] for slot in self._offered]
        selected = self.tree.select_device_writebacks(
            (page.identity for page in candidates), min(self._capacity, len(candidates))
        )
        # The tree accepts one hash path per Host reservation.
        batches = defaultdict(list)
        for _, slot, _ in selected:
            batches[id(self._offered[slot])].append(self.resident[slot])
        for batch in batches.values():
            hashes = self._offered[batch[0].slot]
            plan = self.tree.prepare_host_store(
                hashes, [(page.group, page.row) for page in batch]
            )
            if plan is None:
                continue
            ticket, reservations, _ = plan
            reserved = {index for index, _ in reservations}
            if ticket:
                records = [batch[index] for index, _ in reservations]
                try:
                    self._pin([page.slot for page in records])
                except BaseException:
                    self.tree.settle_host_store(ticket, False)
                    raise
                job_id = next(self._ids)
                self._jobs[job_id] = _Job(
                    False,
                    tuple(
                        (batch[index].group, batch[index].slot, host)
                        for index, host in reservations
                    ),
                    records,
                    ticket,
                )
                self._unsent.append(job_id)
            for index, page in enumerate(batch):
                del self._offered[page.slot]
                if index not in reserved:
                    self._reclaimable.add(page.slot)
        # The tree skips pages that are already present in the Host tier.
        selected_slots = {slot for _, slot, _ in selected}
        for page in candidates:
            if (
                page.slot not in selected_slots
                and self.tree.host_slot(*page.key) is not None
            ):
                del self._offered[page.slot]
                self._reclaimable.add(page.slot)

    def dispatch(self) -> dict[int, tuple[bool, _Copies]]:
        if self._remote is not None:
            self._remote.advance()
        self._schedule_store()
        issued = {
            job_id: (self._jobs[job_id].load, self._jobs[job_id].copies)
            for job_id in self._unsent
        }
        self._unsent.clear()
        return issued

    def settle(
        self, receipts: Mapping[int, Mapping[int, bool]]
    ) -> tuple[set[str], set[str]]:
        finished, failed = set(), set()
        for job_id, ranks in receipts.items():
            job = self._jobs.get(job_id)
            if job is None:
                continue  # A repeated terminal receipt is harmless.
            for rank, success in ranks.items():
                if (
                    not 0 <= rank < self._num_ranks
                    or job.receipts.setdefault(rank, success) != success
                ):
                    raise RuntimeError(
                        f"Conflicting BTC rank {rank} receipt for operation {job_id}"
                    )
            if len(job.receipts) != self._num_ranks:
                continue
            ok = all(job.receipts.values())
            if job.load:
                self._settle_load(job_id, job, ok, finished, failed)
            else:
                if not self.tree.settle_host_store(job.host_ticket, ok):
                    raise RuntimeError(
                        f"Unknown BTC Host reservation {job.host_ticket}"
                    )
                self._unpin([page.slot for page in job.records])
                for page in job.records:
                    if self.resident.get(page.slot) is page:
                        self._reclaimable.add(page.slot)
            del self._jobs[job_id]
        return finished, failed

    def _settle_load(
        self, job_id: int, job: _Job, ok: bool, finished: set[str], failed: set[str]
    ) -> None:
        publish = ok and bool(job.waiters - self._cancelled)
        if not self.tree.settle_device_load(job_id, publish):
            raise RuntimeError(f"Unknown BTC load operation {job_id}")
        self.tree.release_host_load(job.host_ticket)
        for page in job.records:
            del self._incoming[page.key]
            if publish:
                self._publish_loaded(page)
                self.resident[page.slot] = self._by_key[page.key] = page
                self._reclaimable.add(page.slot)
            else:
                self._unpin([page.slot])
        for request_id in job.waiters:
            dependencies = self._dependencies[request_id]
            dependencies.remove(job_id)
            if not ok:
                self._failed.add(request_id)
            if not dependencies:
                finished.add(request_id)
                if request_id in self._failed and request_id not in self._cancelled:
                    failed.add(request_id)
                    self._bypass.add(request_id)
                self._failed.discard(request_id)
                self._cancelled.discard(request_id)
                del self._dependencies[request_id]

    def forget_match(self, request_id: str) -> None:
        if self._remote is not None:
            self._remote.forget(request_id)

    def should_bypass(self, request_id: str) -> bool:
        return request_id in self._bypass

    def finish_request(self, request_id: str) -> None:
        self.forget_match(request_id)
        self._bypass.discard(request_id)
        if request_id in self._dependencies:
            self._cancelled.add(request_id)

    def transfers_pending(self) -> bool:
        return bool(self._jobs or (self._remote is not None and self._remote.busy()))

    def busy(self) -> bool:
        return bool(self._offered) or self.transfers_pending()

    def awaiting_spill(self, ref_count: Callable[[int], int]) -> bool:
        # The next spill can relieve pressure if only the BTC residency hold
        # and possibly the outstanding copy hold are keeping the slot live.
        return any(ref_count(slot) == 1 for slot in self._offered) or any(
            ref_count(page.slot) == 2
            for job in self._jobs.values()
            if not job.load
            for page in job.records
        )

    def reset(self) -> bool:
        if self.resident or self.transfers_pending():
            return False
        if self._remote is not None:
            for request_id in tuple(self._remote.requests):
                self._remote.forget(request_id)
        self.tree.reset_host()
        return True

    def close(self, *, drain_remote: bool) -> None:
        """Release outstanding holds after the caller has stopped copy workers."""
        if self._closed:
            return
        for job_id, job in self._jobs.items():
            if job.load:
                self.tree.settle_device_load(job_id, False)
                self.tree.release_host_load(job.host_ticket)
            else:
                self.tree.settle_host_store(job.host_ticket, False)
            self._unpin([page.slot for page in job.records])
        self._jobs.clear()
        for slot in tuple(self.resident):
            self.invalidate(slot)
        try:
            if self._remote is not None:
                self._remote.close(drain=drain_remote)
                self._remote = None
        finally:
            self.tree.close()
            self._closed = True
