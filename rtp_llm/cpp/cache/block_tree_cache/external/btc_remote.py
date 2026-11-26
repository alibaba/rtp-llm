# SPDX-License-Identifier: Apache-2.0
"""Scheduler-owned leases for asynchronous Host/Tair transfers."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from math import isfinite
from time import monotonic, time

from btc_kvcm import PaceBufferOwnershipUnknown, PaceMetadata, PaceSpec

logger = logging.getLogger(__name__)
Key = tuple[int, bytes]


@dataclass
class RemoteLookup:
    result: Future
    deadline: float
    available: set[Key] | None = None
    # Warm at most once per admission. A dropped or failed object can be recomputed.
    reading: Future | None = None
    finished: bool = False


class BtcHostOwnershipUnknown(RuntimeError):
    """A worker may still be accessing a Host slot after losing its reply."""


class BtcRemoteCache:
    """Only the scheduler thread mutates the tree. I/O borrows pinned Host rows."""

    def __init__(self, tree, storage, config):
        self.lookup_timeout = float(config.get("lookup_timeout_s", 5))
        if not isfinite(self.lookup_timeout) or self.lookup_timeout <= 0:
            raise ValueError("lookup_timeout_s must be finite and positive")
        self.tree = tree
        self.storage = storage
        self.capacity = storage.capacity
        tree.configure_remote_host()
        self.io = ThreadPoolExecutor(max_workers=1, thread_name_prefix="btc-pace")
        self.requests: dict[str, RemoteLookup] = {}
        self.writes: dict[int, Future] = {}
        self.reads: dict[Future, tuple[int, int, set[Key]]] = {}
        self.loading: set[Key] = set()

    @staticmethod
    def _ok(future):
        try:
            return future.result()
        except (BtcHostOwnershipUnknown, PaceBufferOwnershipUnknown):
            raise
        except Exception:
            logger.exception("BTC remote cache operation failed")
            return False

    def advance(self):
        for ticket, future in tuple(self.writes.items()):
            if future.done():
                if not self.tree.settle_remote_write(ticket, bool(self._ok(future))):
                    raise RuntimeError(f"Unknown BTC remote write ticket {ticket}")
                del self.writes[ticket]
        for future, (ticket, lease, keys) in tuple(self.reads.items()):
            if future.done():
                # A lost worker receipt keeps the destination reserved until
                # shutdown. A normal failed read is safe to retire.
                ok = bool(self._ok(future))
                try:
                    if not self.tree.settle_host_store(ticket, ok):
                        raise RuntimeError(f"Unknown BTC restore reservation {ticket}")
                finally:
                    self.tree.release_host_load(lease)
                self.loading.difference_update(keys)
                del self.reads[future]
        if self.writes:
            return
        batch = self.tree.take_remote_writes()
        for index, (ticket, group, key, slot) in enumerate(batch):
            try:
                self.writes[ticket] = self.io.submit(
                    self.storage.write, group, key, slot
                )
            except BaseException:
                for unsent, *_ in batch[index:]:
                    self.tree.settle_remote_write(unsent, False)
                raise

    def warm(self, req_id, hashes, endpoints, device_available):
        """Return False while the request is waiting for its remote prefix."""
        self.advance()
        state = self.requests.get(req_id)
        if state is not None:
            if state.finished:
                return True
            if monotonic() >= state.deadline:
                # A slow write on this peer must not block request admission.
                # Any running read retains its Host reservation until advance().
                state.result.cancel()
                state.finished = True
                return True
            if state.reading is not None:
                if state.reading in self.reads:
                    return False
                state.finished = True
                return True

        def locations(needs):
            absent, host = {}, []
            for group, rows in enumerate(needs):
                for row in rows:
                    key = (group, hashes[row])
                    if device_available(key):
                        continue
                    slot = self.tree.host_slot(*key)
                    if slot is None:
                        absent[key] = (group, row)
                    else:
                        host.append((group, slot))
            return absent, host

        if not endpoints:
            return True
        if state is None:
            if not locations(endpoints[-1][1])[0]:
                return True
            # Full-attention pages are shared by many candidate boundaries.
            # Check the union once, including earlier recurrent checkpoints.
            all_keys = []
            for group in range(len(endpoints[-1][1])):
                # Endpoint ranges grow monotonically. A recurrent group
                # contributes only its final checkpoint at each boundary.
                seen_until = -1
                for _, needs in endpoints:
                    rows = needs[group]
                    if rows:
                        start = max(rows.start, seen_until + rows.step)
                        all_keys.extend(
                            (group, hashes[row])
                            for row in range(start, rows.stop, rows.step)
                            if not device_available((group, hashes[row]))
                        )
                        seen_until = rows[-1]
            lookup = self.io.submit(self.storage.lookup, all_keys)
            self.requests[req_id] = RemoteLookup(
                lookup, monotonic() + self.lookup_timeout
            )
            return False
        if not state.result.done():
            return False
        if state.available is None:
            state.available = self._ok(state.result) or set()
        if not state.available:
            state.finished = True
            return True

        # Prefix sums avoid re-probing a dense prefix at each checkpoint.
        counts = []
        for group, rows in enumerate(endpoints[-1][1]):
            missing = [0]
            host_rows = [0]
            for row in range(rows.step - 1, rows.stop, rows.step):
                key = group, hashes[row]
                needs_host = not device_available(key)
                unavailable = (
                    needs_host
                    and key not in state.available
                    and self.tree.host_slot(*key) is None
                )
                missing.append(missing[-1] + int(unavailable))
                host_rows.append(host_rows[-1] + int(needs_host))
            counts.append((missing, host_rows))

        for _, needs in reversed(endpoints):
            unavailable = 0
            host_rows_needed = 0
            for (missing, host_rows), rows in zip(counts, needs):
                begin, end = rows.start // rows.step, rows.stop // rows.step
                unavailable += missing[end] - missing[begin]
                host_rows_needed += host_rows[end] - host_rows[begin]
            if unavailable or host_rows_needed > self.capacity:
                continue
            absent, host = locations(needs)
            if not absent:
                break
            if (
                not absent.keys() <= state.available
                or len(absent) + len(host) > self.capacity
            ):
                continue
            if absent.keys() & self.loading:
                return False
            lease = self.tree.pin_host_load(host)
            retained = False
            try:
                plan = self.tree.prepare_host_store(hashes, absent.values(), req_id)
                if plan is None:
                    # A previously admitted remote eviction or local DMA may own
                    # the required slots. Fall back if there is no remote progress.
                    if self.tree.remote_has_pending_work():
                        return False
                    continue
                ticket, reservations, _ = plan
                if not ticket:
                    continue
                offered = list(absent)
                items = [(*offered[index], slot) for index, slot in reservations]
                try:
                    future = self.io.submit(self.storage.read_many, items)
                except BaseException:
                    self.tree.settle_host_store(ticket, False)
                    raise
                keys = {offered[index] for index, _ in reservations}
                self.reads[future] = ticket, lease, keys
                self.loading.update(keys)
                state.reading = future
                retained = True
                return False
            finally:
                if not retained:
                    self.tree.release_host_load(lease)
        state.finished = True
        return True

    def forget(self, req_id):
        state = self.requests.pop(req_id, None)
        if state is not None:
            state.result.cancel()
        # A running read still owns its reservation in self.reads.

    def busy(self):
        return bool(self.reads or self.writes or self.tree.remote_has_pending_work())

    def _abandon_transfers(self):
        for ticket, lease, _ in self.reads.values():
            try:
                self.tree.settle_host_store(ticket, False)
            finally:
                self.tree.release_host_load(lease)
        self.reads.clear()
        self.loading.clear()
        for ticket in self.writes:
            self.tree.settle_remote_write(ticket, False)
        self.writes.clear()
        for ticket, *_ in self.tree.take_remote_writes():
            self.tree.settle_remote_write(ticket, False)

    def close(self, *, drain=True):
        for req_id in tuple(self.requests):
            self.forget(req_id)
        try:
            while drain and self.busy():
                self.advance()
                outstanding = [*self.reads, *self.writes.values()]
                if outstanding:
                    wait(outstanding)
        finally:
            for future in (*self.reads, *self.writes.values()):
                future.cancel()
            try:
                self.io.submit(self.storage.close).result()
                # The storage executor has finished accessing the Host slots.
                self._abandon_transfers()
            finally:
                self.io.shutdown(wait=True)


def shard_spec(group: int, rank: int, size: int) -> PaceSpec:
    return PaceSpec(f"G{group}.R{rank}", 0, (size,))


class BtcShardedPaceStorage:
    """Scheduler-side KVCM control and rank-local PACE data plane."""

    def __init__(self, config, group_bytes_by_rank, clients, capacity):
        self.capacity = capacity
        self._group_specs = {
            group: {
                rank: shard_spec(group, rank, size)
                for rank, size in enumerate(sizes)
                if size
            }
            for group, sizes in enumerate(group_bytes_by_rank)
        }
        config = dict(config)
        self._timeout = float(config.get("worker_timeout_s", 30))
        write_timeout = config.setdefault("write_timeout_seconds", 60)
        request_timeout = float(config.get("request_timeout_s", 5))
        sdk_put_ms = (
            config.get("sdk_config", {})
            .get("timeout_config", {})
            .get("put_timeout_ms", 10000)
        )
        if not isfinite(self._timeout) or self._timeout <= 0:
            raise ValueError("BTC PACE worker_timeout_s must be finite and positive")
        if (
            type(write_timeout) is not int
            or type(sdk_put_ms) is not int
            or not isfinite(request_timeout)
            or request_timeout <= 0
            or sdk_put_ms <= 0
            or write_timeout <= request_timeout + self._timeout + sdk_put_ms / 1000
        ):
            raise ValueError(
                "BTC PACE write lease must outlive RPC and SDK write timeouts"
            )
        self._metadata = PaceMetadata(
            config,
            {
                group: tuple(specs.values())
                for group, specs in self._group_specs.items()
            },
        )
        self._clients = dict(clients)
        self._runner = asyncio.Runner()
        self._connected = False

    def _call(self, commands):
        async def fanout():
            return await asyncio.gather(
                *(
                    self._clients[rank].call(command, self._timeout)
                    for rank, command in commands.items()
                ),
                return_exceptions=True,
            )

        results = self._runner.run(fanout())
        for rank, result in zip(commands, results, strict=True):
            if isinstance(result, BaseException):
                # A dropped reply does not prove the rank stopped using its slot.
                raise BtcHostOwnershipUnknown(
                    f"Lost BTC PACE receipt from rank {rank}"
                ) from result
        return dict(zip(commands, results, strict=True))

    def _connect(self):
        if not self._connected:
            storage, options = self._metadata.connect()
            command = {"op": "connect", "storage_configs": storage, "options": options}
            results = self._call(dict.fromkeys(self._clients, command))
            if any(result["failed"] for result in results.values()):
                raise RuntimeError(
                    "A BTC worker did not register its PACE Host mapping"
                )
            self._connected = True

    def lookup(self, keys):
        return self._metadata.lookup(keys)

    def read_many(self, items):
        self._connect()
        requests = [
            ((group, key), spec)
            for group, key, _ in items
            for spec in self._group_specs[group].values()
        ]
        proofs = self._metadata.proofs(requests)
        if len(proofs) != len(requests):
            return False
        by_rank = defaultdict(list)
        for group, key, slot in items:
            for rank, spec in self._group_specs[group].items():
                uri, digests = proofs[(group, key), spec.name]
                by_rank[rank].append((group, slot, uri, digests))
        results = self._call(
            {
                rank: {"op": "read", "entries": entries}
                for rank, entries in by_rank.items()
            }
        )
        return all(not result["failed"] for result in results.values())

    def write(self, group, key, slot):
        self._connect()
        specs = self._group_specs[group]

        def write_shards(uris):
            admit_until = time() + self._timeout
            results = self._call(
                {
                    rank: {
                        "op": "write",
                        "group": group,
                        "slot": slot,
                        "uri": uris[spec.name],
                        "admit_until": admit_until,
                    }
                    for rank, spec in specs.items()
                    if spec.name in uris
                }
            )
            return {
                specs[rank].name: result["digests"]
                for rank, result in results.items()
                if not result["failed"]
            }

        return self._metadata.write_group_via((group, key), write_shards)

    def close(self):
        async def close_clients():
            await asyncio.gather(*(client.close() for client in self._clients.values()))

        try:
            self._runner.run(close_clients())
        finally:
            try:
                self._runner.close()
            finally:
                self._metadata.close()
