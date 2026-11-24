"""Bindings for the scheduler-facing native BlockTreeCache match."""

from __future__ import annotations

import ctypes
import os
import weakref
from collections.abc import Iterable, Sequence
from pathlib import Path

DEVICE = 0
HOST = 1
DISK = 2

READY = 0
JOIN = 1
LOAD = 2

Binding = tuple[int, int, int, int]  # group, hash row, tier, physical slot
JoinBinding = tuple[int, int, int, int, int, int, int]
DevicePublication = tuple[int, int, int, int]  # group, hash row, slot, generation
DeviceResidency = tuple[int, int, int]  # group, slot, generation
DiskResidency = tuple[int, int]  # group, logical disk slot
DeviceLoadTarget = tuple[int, int, int, int, int, int]
Endpoint = tuple[int, Sequence[Sequence[int]]]


class _Key(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("size", ctypes.c_size_t)]


class _PageGroup(ctypes.Structure):
    _fields_ = [("block_size", ctypes.c_size_t), ("last_checkpoint", ctypes.c_bool)]


def _as_key(key: bytes) -> _Key:
    if not isinstance(key, bytes):
        raise TypeError("BTC cache keys must be bytes")
    return _Key(ctypes.cast(ctypes.c_char_p(key), ctypes.c_void_p), len(key))


def _as_array(keys: tuple[bytes, ...]) -> ctypes.Array:
    return (_Key * len(keys))(*map(_as_key, keys))


class _HostResidency(ctypes.Structure):
    _fields_ = [("group", ctypes.c_size_t), ("slot", ctypes.c_int32)]


class _HostStoreCandidate(ctypes.Structure):
    _fields_ = [("group", ctypes.c_size_t), ("row", ctypes.c_size_t)]


class _HostStoreReservation(ctypes.Structure):
    _fields_ = [("candidate", ctypes.c_size_t), ("slot", ctypes.c_int32)]


class _HostStoreVictim(ctypes.Structure):
    _fields_ = [("group", ctypes.c_size_t), ("key", _Key), ("slot", ctypes.c_int32)]


class _RemoteHostWrite(ctypes.Structure):
    _fields_ = [
        ("ticket", ctypes.c_uint64),
        ("group", ctypes.c_size_t),
        ("hash", _Key),
        ("slot", ctypes.c_int32),
    ]


class _HostStoreView(ctypes.Structure):
    _fields_ = [
        ("ticket", ctypes.c_uint64),
        ("reservations", ctypes.POINTER(_HostStoreReservation)),
        ("reservation_count", ctypes.c_size_t),
        ("evicted", ctypes.POINTER(_HostStoreVictim)),
        ("evicted_count", ctypes.c_size_t),
    ]


class _Need(ctypes.Structure):
    _fields_ = [("group", ctypes.c_size_t), ("row", ctypes.c_size_t)]


class _Endpoint(ctypes.Structure):
    _fields_ = [
        ("tokens", ctypes.c_size_t),
        ("needs", ctypes.POINTER(_Need)),
        ("count", ctypes.c_size_t),
    ]


class _Binding(ctypes.Structure):
    _fields_ = [
        ("group", ctypes.c_size_t),
        ("row", ctypes.c_size_t),
        ("tier", ctypes.c_int),
        ("slot", ctypes.c_int32),
    ]

    def as_tuple(self) -> Binding:
        return self.group, self.row, self.tier, self.slot


class _MatchView(ctypes.Structure):
    _fields_ = [
        ("device_tokens", ctypes.c_size_t),
        ("total_tokens", ctypes.c_size_t),
        ("device_bindings", ctypes.POINTER(_Binding)),
        ("num_device_bindings", ctypes.c_size_t),
        ("bindings", ctypes.POINTER(_Binding)),
        ("num_bindings", ctypes.c_size_t),
    ]


class _JoinBinding(ctypes.Structure):
    _fields_ = [
        ("group", ctypes.c_size_t),
        ("row", ctypes.c_size_t),
        ("kind", ctypes.c_int),
        ("tier", ctypes.c_int),
        ("slot", ctypes.c_int32),
        ("generation", ctypes.c_uint64),
        ("operation", ctypes.c_uint64),
    ]

    def as_tuple(self) -> JoinBinding:
        return (
            self.group,
            self.row,
            self.kind,
            self.tier,
            self.slot,
            self.generation,
            self.operation,
        )


class _JoinMatchView(ctypes.Structure):
    _fields_ = [
        ("device_tokens", ctypes.c_size_t),
        ("total_tokens", ctypes.c_size_t),
        ("device_bindings", ctypes.POINTER(_JoinBinding)),
        ("num_device_bindings", ctypes.c_size_t),
        ("bindings", ctypes.POINTER(_JoinBinding)),
        ("num_bindings", ctypes.c_size_t),
    ]


class _DeviceLoadTarget(ctypes.Structure):
    _fields_ = [
        ("group", ctypes.c_size_t),
        ("row", ctypes.c_size_t),
        ("source_tier", ctypes.c_int),
        ("source_slot", ctypes.c_int32),
        ("target_slot", ctypes.c_int32),
        ("generation", ctypes.c_uint64),
    ]


class _DevicePublication(ctypes.Structure):
    _fields_ = [
        ("group", ctypes.c_size_t),
        ("row", ctypes.c_size_t),
        ("slot", ctypes.c_int32),
        ("generation", ctypes.c_uint64),
    ]


class _DeviceResidency(ctypes.Structure):
    _fields_ = [
        ("group", ctypes.c_size_t),
        ("slot", ctypes.c_int32),
        ("generation", ctypes.c_uint64),
    ]

    def as_tuple(self) -> DeviceResidency:
        return self.group, self.slot, self.generation


class _ReclaimView(ctypes.Structure):
    _fields_ = [
        ("transaction", ctypes.c_uint64),
        ("selected", ctypes.POINTER(_DeviceResidency)),
        ("count", ctypes.c_size_t),
    ]

class _DiskResidency(ctypes.Structure):
    _fields_ = [("group", ctypes.c_size_t), ("slot", ctypes.c_int32)]

    def as_tuple(self) -> DiskResidency:
        return self.group, self.slot


class _DiskReclaimView(ctypes.Structure):
    _fields_ = [
        ("has_victim", ctypes.c_bool),
        ("victim", _DiskResidency),
        ("order", ctypes.POINTER(_DiskResidency)),
        ("count", ctypes.c_size_t),
    ]


class _DiskTxnPage(ctypes.Structure):
    _fields_ = [
        ("group", ctypes.c_size_t),
        ("row", ctypes.c_size_t),
        ("slot", ctypes.c_int32),
        ("host_slot", ctypes.c_int32),
        ("active", ctypes.c_bool),
    ]

    def as_dict(self, *, disk: bool, lru: bool = False) -> dict:
        result = {
            "group": self.group,
            "hash_row": self.row,
            "tier": "DISK" if disk else "HOST",
            "source_id": self.slot,
        }
        if self.host_slot >= 0:
            result["host_slot"] = self.host_slot
        if lru:
            result.update(logical_slot=self.slot, active=self.active)
        return result


class _DiskTxnJob(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint64),
        ("disk_ids", ctypes.POINTER(ctypes.c_int32)),
        ("host_ids", ctypes.POINTER(ctypes.c_int32)),
        ("count", ctypes.c_size_t),
    ]

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "disk_ids": self.disk_ids[: self.count],
            "host_ids": self.host_ids[: self.count] if self.host_ids else [],
        }


class _DiskTxnReclaim(ctypes.Structure):
    _fields_ = [
        ("request", _Key),
        ("incoming", _DiskTxnPage),
        ("order", ctypes.POINTER(_DiskTxnPage)),
        ("count", ctypes.c_size_t),
        ("has_victim", ctypes.c_bool),
        ("victim", _DiskTxnPage),
        ("slots_before", ctypes.c_size_t),
        ("slots_after", ctypes.c_size_t),
    ]

    def as_dict(self) -> dict:
        return {
            "request": ctypes.string_at(self.request.data, self.request.size).decode(),
            "incoming": self.incoming.as_dict(disk=False),
            "lru": [page.as_dict(disk=True, lru=True) for page in self.order[: self.count]],
            "victim": self.victim.as_dict(disk=True, lru=True) if self.has_victim else None,
            "slots_before": self.slots_before,
            "slots_after": self.slots_after,
        }


class _DiskTxnJobsView(ctypes.Structure):
    _fields_ = [
        ("jobs", ctypes.POINTER(_DiskTxnJob)),
        ("job_count", ctypes.c_size_t),
        ("reclaims", ctypes.POINTER(_DiskTxnReclaim)),
        ("reclaim_count", ctypes.c_size_t),
    ]


class _DiskTxnEvent(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int),
        ("request", _Key),
        ("pages", ctypes.POINTER(_DiskTxnPage)),
        ("page_count", ctypes.c_size_t),
        ("disk_ids", ctypes.POINTER(ctypes.c_int32)),
        ("disk_id_count", ctypes.c_size_t),
    ]

    def as_dict(self) -> dict:
        return {
            "kind": ("commit", "retry", "failed")[self.kind],
            "request": ctypes.string_at(self.request.data, self.request.size).decode(),
            "pages": [page.as_dict(disk=True) for page in self.pages[: self.page_count]],
            "disk_ids": self.disk_ids[: self.disk_id_count],
        }


class _DiskTxnSettlementView(ctypes.Structure):
    _fields_ = [
        ("done", ctypes.c_bool),
        ("events", ctypes.POINTER(_DiskTxnEvent)),
        ("count", ctypes.c_size_t),
    ]


def _load_library(path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(path)
    handle = ctypes.c_void_p
    key_array = (ctypes.POINTER(_Key), ctypes.c_size_t)
    device_array = (ctypes.POINTER(_DeviceResidency), ctypes.c_size_t)
    disk_array = (ctypes.POINTER(_DiskResidency), ctypes.c_size_t)
    status = ctypes.c_int
    signatures = {
        "last_error": (ctypes.c_char_p, ()),
        "create": (handle, (ctypes.c_size_t,)),
        "destroy": (None, (handle,)),
        "publish": (
            status,
            (handle, *key_array, ctypes.POINTER(_Binding), ctypes.c_size_t),
        ),
        "unpublish": (status, (handle, ctypes.c_size_t, ctypes.c_int, ctypes.c_int32)),
        "configure_host": (status, (handle, ctypes.c_size_t)),
        "prepare_host_store": (
            status,
            (
                handle,
                *key_array,
                ctypes.POINTER(_HostStoreCandidate),
                ctypes.c_size_t,
                _Key,
                ctypes.POINTER(_HostStoreView),
                ctypes.POINTER(handle),
            ),
        ),
        "destroy_host_store_plan": (None, (handle,)),
        "settle_host_store": (status, (handle, ctypes.c_uint64, ctypes.c_bool)),
        "pin_host_load": (
            status,
            (
                handle,
                ctypes.POINTER(_HostResidency),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint64),
            ),
        ),
        "release_host_load": (status, (handle, ctypes.c_uint64)),
        "host_slot": (
            status,
            (handle, ctypes.c_size_t, _Key, ctypes.POINTER(ctypes.c_int32)),
        ),
        "reset_host": (status, (handle,)),
        "publish_device": (
            status,
            (handle, *key_array, ctypes.POINTER(_DevicePublication), ctypes.c_size_t),
        ),
        "unpublish_device": (
            status,
            (handle, ctypes.c_size_t, ctypes.c_int32, ctypes.c_uint64),
        ),
        "begin_device_load": (
            status,
            (
                handle,
                *key_array,
                ctypes.POINTER(_DeviceLoadTarget),
                ctypes.c_size_t,
                ctypes.c_uint64,
            ),
        ),
        "settle_device_load": (status, (handle, ctypes.c_uint64, ctypes.c_bool)),
        "select_device_writebacks": (
            status,
            (
                handle,
                *device_array,
                ctypes.c_size_t,
                ctypes.POINTER(_DeviceResidency),
                ctypes.POINTER(ctypes.c_size_t),
            ),
        ),
        "plan_device_reclaim": (
            status,
            (
                handle,
                *device_array,
                ctypes.c_size_t,
                ctypes.POINTER(_ReclaimView),
                ctypes.POINTER(handle),
            ),
        ),
        "destroy_reclaim_plan": (None, (handle,)),
        "settle_device_reclaim": (status, (handle, ctypes.c_uint64, *device_array)),
        "configure_disk": (status, (handle, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_bool)),
        "pin_disk_load": (status, (handle, *disk_array, ctypes.POINTER(ctypes.c_uint64))),
        "complete_disk_load": (status, (handle, ctypes.c_uint64, ctypes.c_bool)),
        "retry_disk_writes": (status, (handle, *disk_array, ctypes.POINTER(_Key))),
        "take_disk_jobs": (status, (handle, ctypes.POINTER(_DiskTxnJobsView), ctypes.POINTER(handle))),
        "destroy_disk_jobs": (None, (handle,)),
        "settle_disk_job": (
            status,
            (
                handle,
                ctypes.c_uint64,
                ctypes.c_size_t,
                ctypes.c_bool,
                ctypes.POINTER(_DiskTxnSettlementView),
                ctypes.POINTER(handle),
            ),
        ),
        "destroy_disk_settlement": (None, (handle,)),
        "disk_has_pending_work": (status, (handle,)),
        "disk_reset_blocked": (status, (handle,)),
        "reset_disk": (status, (handle,)),
        "touch_disk": (status, (handle, *disk_array)),
        "reclaim_disk": (
            status,
            (handle, *disk_array, ctypes.c_bool, ctypes.POINTER(_DiskReclaimView), ctypes.POINTER(handle)),
        ),
        "destroy_disk_reclaim": (None, (handle,)),
        "match": (
            status,
            (
                handle,
                *key_array,
                ctypes.POINTER(_Endpoint),
                ctypes.c_size_t,
                ctypes.POINTER(_MatchView),
                ctypes.POINTER(handle),
            ),
        ),
        "destroy_match": (None, (handle,)),
        "match_with_join": (
            status,
            (
                handle,
                *key_array,
                ctypes.POINTER(_Endpoint),
                ctypes.c_size_t,
                ctypes.POINTER(_JoinMatchView),
                ctypes.POINTER(handle),
            ),
        ),
        "destroy_join_match": (None, (handle,)),
        "reset": (status, (handle,)),
    }
    _bind(lib, signatures)
    return lib


def _bind(lib, signatures):
    for name, (restype, argtypes) in signatures.items():
        function = getattr(lib, f"btc_tree_{name}")
        function.restype = restype
        function.argtypes = argtypes


def _bind_remote(lib):
    handle = ctypes.c_void_p
    status = ctypes.c_int
    _bind(lib, {
        "configure_remote_host": (status, (handle,)),
        "remote_has_pending_work": (status, (handle,)),
        "take_remote_writes": (
            status,
            (handle, ctypes.POINTER(ctypes.POINTER(_RemoteHostWrite)),
             ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(handle)),
        ),
        "destroy_remote_writes": (None, (handle,)),
        "settle_remote_write": (status, (handle, ctypes.c_uint64, ctypes.c_bool)),
        "check_remote_close": (status, (handle,)),
    })


class TreeCache:
    """Tree of acknowledged external slots borrowing the engine's lifetime."""

    DEVICE = DEVICE
    HOST = HOST
    DISK = DISK
    READY = READY
    JOIN = JOIN
    LOAD = LOAD

    def __init__(self, group_count: int, library_path: str | None = None):
        if not 0 < group_count < 2**31 - 1:
            raise ValueError("BTC group count must be positive")
        path = library_path or os.getenv("RTP_BTC_TREE_LIBRARY")
        if not path:
            bundled = Path(__file__).with_name("_btc_tpu_lib") / "libbtc_tree.so"
            if not bundled.is_file():
                raise ValueError("Set RTP_BTC_TREE_LIBRARY to the path to libbtc_tree.so")
            path = str(bundled)
        self._group_count = group_count
        self._lib = _load_library(path)
        self._remote_host = False
        self._whole_page_match = None
        self._native_handle = self._lib.btc_tree_create(group_count)
        if not self._native_handle:
            self._raise_native_error()
        self._destroy = weakref.finalize(self, self._lib.btc_tree_destroy, self._native_handle)

    @property
    def _handle(self):
        if self._native_handle is None:
            raise RuntimeError("BTC tree is closed")
        return self._native_handle

    def close(self) -> None:
        if self._native_handle is None:
            return
        if self._remote_host:
            self._check(self._lib.btc_tree_check_remote_close(self._handle))
        self._destroy()
        self._native_handle = None

    def _raise_native_error(self) -> None:
        message = self._lib.btc_tree_last_error().decode(errors="replace")
        raise RuntimeError(message)

    def _check(self, result: int) -> None:
        if result < 0:
            self._raise_native_error()

    def _validate_slot(self, group: int, tier: int, slot: int) -> None:
        if not (
            0 <= group < self._group_count
            and tier in (DEVICE, HOST, DISK)
            and 0 <= slot < 2**31
        ):
            raise ValueError("BTC publication has an invalid group, tier or slot")

    def _validate_device(
        self, group: int, slot: int, generation: int, *, allow_unguarded: bool = False
    ) -> None:
        self._validate_slot(group, DEVICE, slot)
        min_generation = 0 if allow_unguarded else 1
        if slot == 0 or not min_generation <= generation < 2**64:
            raise ValueError("BTC device residency has an invalid slot or generation")

    def _device_array(
        self, residencies: Iterable[DeviceResidency], *, allow_unguarded: bool = False
    ) -> ctypes.Array:
        residencies = tuple(residencies)
        for residency in residencies:
            self._validate_device(*residency, allow_unguarded=allow_unguarded)
        return (_DeviceResidency * len(residencies))(
            *(_DeviceResidency(*residency) for residency in residencies)
        )

    def _disk_array(self, pages: Iterable[DiskResidency]) -> ctypes.Array:
        pages = tuple(pages)
        for group, slot in pages:
            self._validate_slot(group, DISK, slot)
        return (_DiskResidency * len(pages))(*(_DiskResidency(*page) for page in pages))

    def configure_host(self, capacity: int) -> None:
        if not 0 < capacity < 2**31 - 1:
            raise ValueError("invalid BTC Host row capacity")
        self._check(self._lib.btc_tree_configure_host(self._handle, capacity))

    def configure_remote_host(self) -> None:
        _bind_remote(self._lib)
        self._check(self._lib.btc_tree_configure_remote_host(self._handle))
        self._remote_host = True

    def remote_has_pending_work(self) -> bool:
        if not self._remote_host:
            return False
        status = self._lib.btc_tree_remote_has_pending_work(self._handle)
        self._check(status)
        return bool(status)

    def take_remote_writes(self) -> list[tuple[int, int, bytes, int]]:
        if not self._remote_host:
            return []
        rows = ctypes.POINTER(_RemoteHostWrite)()
        count = ctypes.c_size_t()
        snapshot = ctypes.c_void_p()
        self._check(self._lib.btc_tree_take_remote_writes(
            self._handle, ctypes.byref(rows), ctypes.byref(count), ctypes.byref(snapshot)
        ))
        try:
            return [
                (row.ticket, row.group, ctypes.string_at(row.hash.data, row.hash.size), row.slot)
                for row in rows[:count.value]
            ]
        except BaseException:
            # No Python caller has received these tickets or started I/O.
            for index in range(count.value):
                self.settle_remote_write(rows[index].ticket, False)
            raise
        finally:
            self._lib.btc_tree_destroy_remote_writes(snapshot)

    def settle_remote_write(self, ticket: int, success: bool) -> bool:
        if not 0 < ticket < 2**64:
            raise ValueError("invalid BTC Host writeback ticket")
        if not self._remote_host:
            return False
        status = self._lib.btc_tree_settle_remote_write(self._handle, ticket, success)
        self._check(status)
        return bool(status)

    def prepare_host_store(
        self,
        hashes: Sequence[bytes],
        candidates: Iterable[tuple[int, int]],
        request_id: str = "",
    ) -> tuple[int, tuple[tuple[int, int], ...], tuple[tuple[int, bytes, int], ...]] | None:
        hashes = tuple(hashes)
        path = _as_array(hashes)
        offered = tuple(candidates)
        for group, row in offered:
            if not (0 <= group < self._group_count and 0 <= row < len(hashes)):
                raise ValueError("invalid BTC Host store candidate")
        records = (_HostStoreCandidate * len(offered))(
            *(_HostStoreCandidate(*candidate) for candidate in offered)
        )
        view = _HostStoreView()
        plan = ctypes.c_void_p()
        status = self._lib.btc_tree_prepare_host_store(
            self._handle,
            path,
            len(path),
            records,
            len(records),
            _as_key(request_id.encode()),
            ctypes.byref(view),
            ctypes.byref(plan),
        )
        self._check(status)
        if not status:
            return None
        try:
            reservations = tuple(
                (slot.candidate, slot.slot)
                for slot in view.reservations[: view.reservation_count]
            )
            evicted = tuple(
                (row.group, ctypes.string_at(row.key.data, row.key.size), row.slot)
                for row in view.evicted[: view.evicted_count]
            )
            return view.ticket, reservations, evicted
        finally:
            self._lib.btc_tree_destroy_host_store_plan(plan)

    def settle_host_store(self, ticket: int, success: bool) -> bool:
        status = self._lib.btc_tree_settle_host_store(self._handle, ticket, success)
        self._check(status)
        return bool(status)

    def pin_host_load(self, pages: Iterable[tuple[int, int]]) -> int:
        pages = tuple(pages)
        for group, slot in pages:
            self._validate_slot(group, HOST, slot)
        records = (_HostResidency * len(pages))(
            *(_HostResidency(*page) for page in pages)
        )
        ticket = ctypes.c_uint64()
        self._check(
            self._lib.btc_tree_pin_host_load(
                self._handle, records, len(records), ctypes.byref(ticket)
            )
        )
        return ticket.value

    def release_host_load(self, ticket: int) -> bool:
        status = self._lib.btc_tree_release_host_load(self._handle, ticket)
        self._check(status)
        return bool(status)

    def host_slot(self, group: int, hash_bytes: bytes) -> int | None:
        if not 0 <= group < self._group_count:
            raise ValueError("invalid BTC Host group")
        slot = ctypes.c_int32()
        status = self._lib.btc_tree_host_slot(
            self._handle, group, _as_key(hash_bytes), ctypes.byref(slot)
        )
        self._check(status)
        return slot.value if status else None

    def reset_host(self) -> None:
        self._check(self._lib.btc_tree_reset_host(self._handle))

    def publish(self, keys: Sequence[bytes], bindings: Iterable[Binding]) -> None:
        bindings = list(bindings)
        for group, row, tier, slot in bindings:
            self._validate_slot(group, tier, slot)
            if not 0 <= row < len(keys):
                raise ValueError("BTC publication row must refer to the hash path")
        native_bindings = (_Binding * len(bindings))(
            *(_Binding(*binding) for binding in bindings)
        )
        self._check(
            self._lib.btc_tree_publish(
                self._handle, _as_array(tuple(keys)), len(keys), native_bindings, len(bindings)
            )
        )

    def unpublish(self, group_id: int, tier: int, slot: int) -> None:
        self._validate_slot(group_id, tier, slot)
        self._check(self._lib.btc_tree_unpublish(self._handle, group_id, tier, slot))

    def publish_device(
        self, keys: Sequence[bytes], bindings: Iterable[DevicePublication]
    ) -> None:
        bindings = tuple(bindings)
        for group, row, slot, generation in bindings:
            self._validate_device(group, slot, generation)
            if not 0 <= row < len(keys):
                raise ValueError("BTC device publication row must refer to the hash path")
        native_bindings = (_DevicePublication * len(bindings))(
            *(_DevicePublication(*binding) for binding in bindings)
        )
        self._check(
            self._lib.btc_tree_publish_device(
                self._handle,
                _as_array(tuple(keys)),
                len(keys),
                native_bindings,
                len(bindings),
            )
        )

    def unpublish_device(self, group: int, slot: int, generation: int) -> None:
        self._validate_device(group, slot, generation)
        self._check(
            self._lib.btc_tree_unpublish_device(self._handle, group, slot, generation)
        )

    def begin_device_load(
        self,
        keys: Sequence[bytes],
        targets: Iterable[DeviceLoadTarget],
        operation_id: int,
    ) -> None:
        """Register one submitted producer whose source and targets stay pinned."""
        if not 0 < operation_id < 2**64:
            raise ValueError("BTC load operation must be a positive uint64_t")
        targets = tuple(targets)
        for group, row, tier, source_slot, target_slot, generation in targets:
            self._validate_device(group, target_slot, generation)
            self._validate_slot(group, tier, source_slot)
            if tier not in (HOST, DISK) or not 0 <= row < len(keys):
                raise ValueError("BTC load source must be a Host or Disk hash row")
        native_targets = (_DeviceLoadTarget * len(targets))(
            *(_DeviceLoadTarget(*target) for target in targets)
        )
        self._check(
            self._lib.btc_tree_begin_device_load(
                self._handle,
                _as_array(tuple(keys)),
                len(keys),
                native_targets,
                len(native_targets),
                operation_id,
            )
        )

    def settle_device_load(self, operation_id: int, success: bool) -> bool:
        """Finish a fenced producer. A success requires a retained GPU guard."""
        if not 0 < operation_id < 2**64:
            raise ValueError("BTC load operation must be a positive uint64_t")
        result = self._lib.btc_tree_settle_device_load(
            self._handle, operation_id, success
        )
        self._check(result)
        return bool(result)

    def select_device_writebacks(
        self, eligible: Iterable[DeviceResidency], limit: int
    ) -> tuple[DeviceResidency, ...]:
        """Select live Device-only copies in the order offered by the scheduler."""
        if not 0 <= limit <= ctypes.c_size_t(-1).value:
            raise ValueError("BTC device writeback limit is outside size_t")
        candidates = self._device_array(eligible, allow_unguarded=True)
        output = (_DeviceResidency * min(len(candidates), limit))()
        output_count = ctypes.c_size_t()
        self._check(
            self._lib.btc_tree_select_device_writebacks(
                self._handle,
                candidates,
                len(candidates),
                limit,
                output,
                ctypes.byref(output_count),
            )
        )
        return tuple(output[i].as_tuple() for i in range(output_count.value))

    def plan_device_reclaim(
        self, eligible: Iterable[DeviceResidency], limit: int
    ) -> tuple[int, list[DeviceResidency]]:
        """Reserve up to limit of the eligible GPU guards in BTC eviction order."""
        if not 0 <= limit <= ctypes.c_size_t(-1).value:
            raise ValueError("BTC device reclaim limit is outside size_t")
        candidates = self._device_array(eligible)
        view = _ReclaimView()
        result = ctypes.c_void_p()
        self._check(
            self._lib.btc_tree_plan_device_reclaim(
                self._handle,
                candidates,
                len(candidates),
                limit,
                ctypes.byref(view),
                ctypes.byref(result),
            )
        )
        try:
            return view.transaction, [entry.as_tuple() for entry in view.selected[: view.count]]
        finally:
            self._lib.btc_tree_destroy_reclaim_plan(result)

    def settle_device_reclaim(
        self, transaction: int, released: Iterable[DeviceResidency]
    ) -> None:
        """Drop acknowledged generations and restore all other reservations."""
        if not 0 <= transaction < 2**64:
            raise ValueError("BTC device reclaim transaction is outside uint64_t")
        acknowledged = self._device_array(released)
        self._check(
            self._lib.btc_tree_settle_device_reclaim(
                self._handle, transaction, acknowledged, len(acknowledged)
            )
        )

    def configure_disk(self, capacity: int, num_ranks: int, trace: bool) -> None:
        if not (0 < capacity < 2**31 and 0 < num_ranks <= ctypes.c_size_t(-1).value):
            raise ValueError("invalid BTC Disk capacity or rank count")
        self._check(self._lib.btc_tree_configure_disk(self._handle, capacity, num_ranks, trace))

    def pin_disk_load(self, pages: Iterable[DiskResidency]) -> int:
        source = self._disk_array(pages)
        lease = ctypes.c_uint64()
        self._check(
            self._lib.btc_tree_pin_disk_load(
                self._handle, source, len(source), ctypes.byref(lease)
            )
        )
        return lease.value

    def complete_disk_load(self, lease: int, success: bool) -> None:
        if not 0 <= lease < 2**64:
            raise ValueError("invalid BTC Disk load lease")
        self._check(self._lib.btc_tree_complete_disk_load(self._handle, lease, success))

    def retry_disk_writes(self, host_pages: Iterable[DiskResidency], request_id: str) -> None:
        source = self._disk_array(host_pages)
        request = _as_array((request_id.encode(),))
        self._check(
            self._lib.btc_tree_retry_disk_writes(self._handle, source, len(source), request)
        )

    def take_disk_jobs(self) -> tuple[list[dict], list[dict]]:
        view = _DiskTxnJobsView()
        handle = ctypes.c_void_p()
        self._check(
            self._lib.btc_tree_take_disk_jobs(self._handle, ctypes.byref(view), ctypes.byref(handle))
        )
        try:
            return (
                [job.as_dict() for job in view.jobs[: view.job_count]],
                [event.as_dict() for event in view.reclaims[: view.reclaim_count]],
            )
        finally:
            self._lib.btc_tree_destroy_disk_jobs(handle)

    def settle_disk_job(
        self, job_id: int, completions: int, failed: bool
    ) -> tuple[bool, list[dict]]:
        if not (0 < job_id < 2**64 and 0 < completions <= ctypes.c_size_t(-1).value):
            raise ValueError("invalid BTC Disk completion receipt")
        view = _DiskTxnSettlementView()
        handle = ctypes.c_void_p()
        self._check(
            self._lib.btc_tree_settle_disk_job(
                self._handle,
                job_id,
                completions,
                failed,
                ctypes.byref(view),
                ctypes.byref(handle),
            )
        )
        try:
            return view.done, [event.as_dict() for event in view.events[: view.count]]
        finally:
            self._lib.btc_tree_destroy_disk_settlement(handle)

    def disk_has_pending_work(self) -> bool:
        result = self._lib.btc_tree_disk_has_pending_work(self._handle)
        self._check(result)
        return bool(result)

    def disk_reset_blocked(self) -> bool:
        result = self._lib.btc_tree_disk_reset_blocked(self._handle)
        self._check(result)
        return bool(result)

    def reset_disk(self) -> None:
        self._check(self._lib.btc_tree_reset_disk(self._handle))

    def touch_disk(self, pages: Iterable[DiskResidency]) -> None:
        """Record successful Disk loads, in plan order."""
        records = self._disk_array(pages)
        self._check(self._lib.btc_tree_touch_disk(self._handle, records, len(records)))

    def reclaim_disk(
        self, protected: Iterable[DiskResidency], snapshot: bool = False
    ) -> tuple[DiskResidency | None, list[DiskResidency]]:
        """Withdraw the oldest unprotected Disk page before slot release."""
        held = self._disk_array(protected)
        view = _DiskReclaimView()
        result = ctypes.c_void_p()
        self._check(
            self._lib.btc_tree_reclaim_disk(
                self._handle, held, len(held), snapshot, ctypes.byref(view), ctypes.byref(result)
            )
        )
        try:
            victim = view.victim.as_tuple() if view.has_victim else None
            return victim, [item.as_tuple() for item in view.order[: view.count]]
        finally:
            self._lib.btc_tree_destroy_disk_reclaim(result)

    def _match_endpoints(
        self, keys: Sequence[bytes], endpoints: Sequence[Endpoint]
    ) -> ctypes.Array:
        native_endpoints = (_Endpoint * len(endpoints))()
        for index, (tokens, by_group) in enumerate(endpoints):
            if tokens <= 0 or len(by_group) != self._group_count:
                raise ValueError("BTC endpoint requires positive tokens and all groups")
            needs = [
                (group, row)
                for group, rows in enumerate(by_group)
                for row in rows
            ]
            if any(not 0 <= row < len(keys) for _, row in needs):
                raise ValueError("BTC endpoint need refers to an invalid hash row")
            native_needs = (_Need * len(needs))(*(_Need(*need) for need in needs))
            native_endpoints[index] = _Endpoint(tokens, native_needs, len(needs))
        return native_endpoints

    def match(
        self, keys: Sequence[bytes], endpoints: Sequence[Endpoint]
    ) -> tuple[int, int, list[Binding], list[Binding]]:
        native_endpoints = self._match_endpoints(keys, endpoints)
        view = _MatchView()
        result = ctypes.c_void_p()
        self._check(
            self._lib.btc_tree_match(
                self._handle,
                _as_array(tuple(keys)),
                len(keys),
                native_endpoints,
                len(endpoints),
                ctypes.byref(view),
                ctypes.byref(result),
            )
        )
        try:
            return (
                view.device_tokens,
                view.total_tokens,
                [
                    binding.as_tuple()
                    for binding in view.device_bindings[: view.num_device_bindings]
                ],
                [binding.as_tuple() for binding in view.bindings[: view.num_bindings]],
            )
        finally:
            self._lib.btc_tree_destroy_match(result)

    def match_with_join(
        self, keys: Sequence[bytes], endpoints: Sequence[Endpoint]
    ) -> tuple[int, int, list[JoinBinding], list[JoinBinding]]:
        """Return READY, JOIN and LOAD bindings, with exact target generations."""
        native_endpoints = self._match_endpoints(keys, endpoints)
        view = _JoinMatchView()
        result = ctypes.c_void_p()
        self._check(
            self._lib.btc_tree_match_with_join(
                self._handle,
                _as_array(tuple(keys)),
                len(keys),
                native_endpoints,
                len(native_endpoints),
                ctypes.byref(view),
                ctypes.byref(result),
            )
        )
        return self._read_join_match(view, result)

    def match_whole_pages_with_join(
        self, keys: Sequence[bytes], maximum_tokens: int, hash_step: int,
        group_sizes: Sequence[int], last_checkpoint: Sequence[bool],
    ) -> tuple[int, int, list[JoinBinding], list[JoinBinding]]:
        """Match FULL prefixes and recurrent final states in one native path scan."""
        if not (0 <= maximum_tokens <= ctypes.c_size_t(-1).value and 0 < hash_step <= ctypes.c_size_t(-1).value):
            raise ValueError("invalid BTC whole-page match limit or hash step")
        if len(group_sizes) != self._group_count or len(last_checkpoint) != self._group_count:
            raise ValueError("BTC match needs one page layout per cache group")
        if any(not 0 < size <= ctypes.c_size_t(-1).value or size % hash_step for size in group_sizes):
            raise ValueError("each BTC group must contain whole hash rows")
        if any(type(flag) is not bool for flag in last_checkpoint):
            raise ValueError("BTC final checkpoint selection must be boolean")
        if self._whole_page_match is None:
            fn = self._lib.btc_tree_match_whole_pages_with_join
            fn.restype = ctypes.c_int
            fn.argtypes = (ctypes.c_void_p, ctypes.POINTER(_Key), ctypes.c_size_t,
                           ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(_PageGroup),
                           ctypes.c_size_t, ctypes.POINTER(_JoinMatchView), ctypes.POINTER(ctypes.c_void_p))
            self._whole_page_match = fn
        groups = (_PageGroup * self._group_count)(*map(_PageGroup, group_sizes, last_checkpoint))
        keys = tuple(keys[:maximum_tokens // hash_step])
        view, result = _JoinMatchView(), ctypes.c_void_p()
        self._check(self._whole_page_match(
            self._handle, _as_array(keys), len(keys), maximum_tokens, hash_step,
            groups, len(groups), ctypes.byref(view), ctypes.byref(result),
        ))
        return self._read_join_match(view, result)

    def _read_join_match(self, view, result):
        try:
            return (
                view.device_tokens,
                view.total_tokens,
                [
                    binding.as_tuple()
                    for binding in view.device_bindings[: view.num_device_bindings]
                ],
                [binding.as_tuple() for binding in view.bindings[: view.num_bindings]],
            )
        finally:
            self._lib.btc_tree_destroy_join_match(result)

    def reset(self) -> None:
        self._check(self._lib.btc_tree_reset(self._handle))
