"""Python access to the BlockTreeCache C ABI."""

from __future__ import annotations

import ctypes as _c
import ctypes.util as _util
import os
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

ABI_VERSION = 2
HOST_POOL = (1 << 64) - 1
DISK_POOL = HOST_POOL - 1


class Status(IntEnum):
    OK = 0
    INVALID = 1
    PENDING = 2
    RETRY = 3
    WAITING = 4
    CLOSED = 5
    IO = 6
    INTERNAL = 7


class Rule(IntEnum):
    FULL = 1
    LINEAR = 2
    SWA = 3


class PageState(IntEnum):
    READY = 1
    LOAD = 2
    JOIN = 3


class WorkKind(IntEnum):
    HOST_TO_DEVICE = 1
    DEVICE_TO_HOST = 2
    DISK_TO_HOST = 3
    HOST_TO_DISK = 4
    REMOTE_QUERY = 5
    REMOTE_TO_HOST = 6
    HOST_TO_REMOTE = 7
    RELEASE = 8
    REMOTE_TO_DEVICE = 9
    DEVICE_TO_REMOTE = 10
    DISK_TO_DEVICE = 11


class Result(IntEnum):
    SUCCESS = 1
    FAILED = 2
    TIMEOUT = 3
    REJECTED = 4


class EventKind(IntEnum):
    READY = 1
    RECOMPUTE = 2
    CANCELLED = 3
    PROGRESS = 4


class Error(RuntimeError):
    def __init__(self, status: Status, message: str):
        self.status = status
        super().__init__(message or status.name)


@dataclass(frozen=True)
class Group:
    id: int
    rule: Rule
    page_tokens: int
    window_tokens: int = 0
    members: int = 1
    member_pools: tuple[int, ...] | None = None
    host_slots: int = 0
    disk_slots: int = 0


@dataclass(frozen=True)
class HostRegion:
    """Writable storage shared with the executors, with one payload per row."""

    buffer: object
    stride_bytes: int
    payload_bytes: int


@dataclass(frozen=True)
class Position:
    group: int
    end_token: int


@dataclass(frozen=True)
class Page:
    pool: int
    slot: int
    generation: int
    hold: int = 0


@dataclass(frozen=True)
class Binding:
    position: Position
    pages: tuple[Page, ...]


@dataclass(frozen=True)
class Limits:
    max_tokens: int
    groups: tuple[int, ...] | None = None
    boundaries: tuple[int, ...] | None = None
    max_load_pages: int = _c.c_size_t(-1).value
    allow_remote: bool = True
    max_load_tokens: int | None = None


@dataclass(frozen=True)
class PlanPage:
    position: Position
    state: PageState
    pages: tuple[Page, ...]
    operation: int
    route: WorkKind | None = None


@dataclass(frozen=True)
class Operation:
    instance: int
    operation: int


@dataclass(frozen=True)
class WorkPage:
    position: Position
    key: bytes
    source: tuple[Page, ...]
    target: tuple[Page, ...]
    remote: bytes


@dataclass(frozen=True)
class Work:
    id: Operation
    kind: WorkKind
    request: int
    name_space: bytes
    participants: tuple[int, ...]
    pages: tuple[WorkPage, ...]


@dataclass(frozen=True)
class Event:
    cursor: int
    request: int
    kind: EventKind
    tokens: int


@dataclass(frozen=True)
class Batch:
    work: tuple[Work, ...]
    events: tuple[Event, ...]


@dataclass(frozen=True)
class RemoteObject:
    position: Position
    descriptor: bytes


@dataclass(frozen=True)
class Receipt:
    id: Operation
    participant: int
    result: Result
    safe: bool
    committed: bool = False
    objects: tuple[RemoteObject, ...] = ()


_U32, _U64, _Size, _Ptr, _P = _c.c_uint32, _c.c_uint64, _c.c_size_t, _c.c_void_p, _c.POINTER


class _Bytes(_c.Structure):
    _fields_ = [("data", _Ptr), ("size", _Size)]

    @classmethod
    def encode(cls, value):
        if not isinstance(value, bytes):
            raise TypeError("BTC namespace, keys, and descriptors must be bytes")
        return cls(_c.cast(_c.c_char_p(value), _Ptr), len(value))

    def decode(self):
        return _c.string_at(self.data, self.size) if self.size else b""


class _Keys(_c.Structure):
    _fields_ = [("keys", _P(_Bytes)), ("count", _Size)]


class _Position(_c.Structure):
    _fields_ = [("group", _U32), ("end_token", _U64)]

    def decode(self):
        return Position(self.group, self.end_token)


class _Page(_c.Structure):
    _fields_ = [("pool", _U64), ("slot", _U64), ("generation", _U64), ("hold", _U64)]

    def decode(self):
        return Page(self.pool, self.slot, self.generation, self.hold)


class _Group(_c.Structure):
    _fields_ = [
        ("id", _U32), ("rule", _U32), ("page_tokens", _U64), ("window_tokens", _U64),
        ("members", _Size), ("member_pools", _P(_U64)),
        ("host_slots", _U64), ("disk_slots", _U64),
    ]


class _Route(_c.Structure):
    _fields_ = [("kind", _U32), ("participants", _P(_U32)), ("count", _Size)]


class _Config(_c.Structure):
    _fields_ = [
        ("version", _U32), ("namespace", _Bytes), ("key_tokens", _U64), ("groups", _P(_Group)),
        ("group_count", _Size), ("host_slots", _U64), ("disk_slots", _U64),
        ("disk_directory", _Bytes), ("routes", _P(_Route)), ("route_count", _Size),
        ("release_local_pages", _U32),
        ("device_eviction_policy", _U32),
        ("external_disk_io", _U32), ("tiered_eviction", _U32),
    ]


class _Limits(_c.Structure):
    _fields_ = [
        ("max_tokens", _U64), ("groups", _P(_U32)), ("group_count", _Size),
        ("boundaries", _P(_U64)), ("boundary_count", _Size), ("max_load_pages", _Size),
        ("allow_remote", _U32),
    ]


class _PlanPage(_c.Structure):
    _fields_ = [
        ("position", _Position), ("state", _U32), ("pages", _P(_Page)), ("page_count", _Size),
        ("operation", _U64), ("route", _U32),
    ]

    def decode(self):
        return PlanPage(self.position.decode(), PageState(self.state),
                        _decode(self.pages, self.page_count), self.operation,
                        WorkKind(self.route) if self.route else None)


class _PlanView(_c.Structure):
    _fields_ = [
        ("request", _U64), ("ready_tokens", _U64), ("recoverable_tokens", _U64),
        ("pages", _P(_PlanPage)), ("page_count", _Size),
        ("ready_pages", _P(_PlanPage)), ("ready_page_count", _Size),
        ("host_tokens", _U64),
    ]


class _Binding(_c.Structure):
    _fields_ = [("position", _Position), ("pages", _P(_Page)), ("page_count", _Size)]


class _Operation(_c.Structure):
    _fields_ = [("instance", _U64), ("operation", _U64)]

    def decode(self):
        return Operation(self.instance, self.operation)


class _WorkPage(_c.Structure):
    _fields_ = [
        ("position", _Position), ("key", _Bytes), ("source", _P(_Page)), ("source_count", _Size),
        ("target", _P(_Page)), ("target_count", _Size), ("remote", _Bytes),
    ]

    def decode(self):
        return WorkPage(self.position.decode(), self.key.decode(),
                        _decode(self.source, self.source_count), _decode(self.target, self.target_count),
                        self.remote.decode())


class _Work(_c.Structure):
    _fields_ = [
        ("id", _Operation), ("kind", _U32), ("request", _U64), ("name_space", _Bytes),
        ("participants", _P(_U32)), ("participant_count", _Size),
        ("pages", _P(_WorkPage)), ("page_count", _Size),
    ]

    def decode(self):
        return Work(self.id.decode(), WorkKind(self.kind), self.request, self.name_space.decode(),
                    tuple(self.participants[:self.participant_count]), _decode(self.pages, self.page_count))


class _Event(_c.Structure):
    _fields_ = [("cursor", _U64), ("request", _U64), ("kind", _U32), ("tokens", _U64)]

    def decode(self):
        return Event(self.cursor, self.request, EventKind(self.kind), self.tokens)


class _BatchView(_c.Structure):
    _fields_ = [("work", _P(_Work)), ("work_count", _Size), ("events", _P(_Event)), ("event_count", _Size)]


class _RemoteObject(_c.Structure):
    _fields_ = [("position", _Position), ("descriptor", _Bytes)]


class _Receipt(_c.Structure):
    _fields_ = [
        ("id", _Operation), ("participant", _U32), ("result", _U32), ("safe", _U32),
        ("committed", _U32), ("objects", _P(_RemoteObject)), ("object_count", _Size),
    ]


class _Buffer(_c.Structure):
    _fields_ = [("data", _Ptr), ("size", _Size)]


class _HostRegion(_c.Structure):
    _fields_ = [("memory", _Buffer), ("stride_bytes", _Size), ("payload_bytes", _Size)]


def _load_library(path, *, with_host_region=False):
    if path is None:
        path = os.environ.get("BLOCK_TREE_CACHE_LIBRARY")
    if path is None:
        directory = Path(__file__).resolve().parent
        path = next((directory / name for name in (
            "libblock_tree_cache.so", "libblock_tree_cache.dylib", "block_tree_cache.dll"
        ) if (directory / name).is_file()), None)
    if path is None:
        path = _util.find_library("block_tree_cache")
    if path is None:
        raise OSError("Install libblock_tree_cache or set BLOCK_TREE_CACHE_LIBRARY")
    lib = _c.CDLL(os.fspath(path))
    lib.btc_abi_version.argtypes = []
    lib.btc_abi_version.restype = _U32
    if lib.btc_abi_version() != ABI_VERSION:
        raise OSError(f"Unsupported BlockTreeCache ABI {lib.btc_abi_version()}")
    signatures = {
        "last_error": (_c.c_char_p, []),
        "open": (_U32, [_P(_Config), _P(_Ptr)]),
        "match": (_U32, [_Ptr, _U64, _Keys, _P(_Limits), _P(_Ptr)]),
        "match_capped": (_U32, [_Ptr, _U64, _Keys, _P(_Limits), _U64, _P(_Ptr)]),
        "poll_match": (_U32, [_Ptr, _U64, _P(_Ptr)]),
        "plan_get": (_PlanView, [_Ptr]),
        "plan_free": (None, [_Ptr]),
        "bind": (_U32, [_Ptr, _Ptr, _P(_Binding), _Size]),
        "publish": (_U32, [_Ptr, _Keys, _U64, _P(_Binding), _Size]),
        "poll": (_U32, [_Ptr, _P(_Ptr)]),
        "pending_work": (_U32, [_Ptr, _P(_Size)]),
        "batch_get": (_BatchView, [_Ptr]),
        "batch_free": (None, [_Ptr]),
        "accept": (_U32, [_Ptr, _Operation, _U32]),
        "complete": (_U32, [_Ptr, _P(_Receipt), _Size]),
        "ack_events": (_U32, [_Ptr, _U64]),
        "reclaim": (_U32, [_Ptr, _U64, _Size, _P(_Page), _Size]),
        "host_capacity": (_U32, [_Ptr, _U32, _P(_Size), _P(_Size)]),
        "installed_device_pages": (_U32, [_Ptr, _Keys, _P(_Binding), _Size, _P(_c.c_uint8)]),
        "device_balance_needed": (_U32, [_Ptr, _U64, _Size, _Size, _P(_c.c_uint8)]),
        "balance_device": (_U32, [_Ptr, _U64, _Size, _Size, _P(_Page), _Size]),
        "demote": (_U32, [_Ptr, _U64, _Size, _P(_Page), _Size, _U32]),
        "invalidate": (_U32, [_Ptr, _P(_Page), _Size]),
        "cancel": (_U32, [_Ptr, _U64]),
        "drain": (_U32, [_Ptr]),
        "close": (_U32, [_Ptr]),
        "execute_disk": (_U32, [_Ptr, _Operation, _P(_Buffer), _Size]),
    }
    if with_host_region:
        signatures["open_with_host_region"] = (_U32, [_P(_Config), _P(_HostRegion), _P(_Ptr)])
        signatures["host_slot_buffer"] = (_U32, [_Ptr, _U32, _U64, _P(_Buffer)])
    for name, (result, arguments) in signatures.items():
        method = getattr(lib, f"btc_{name}")
        method.restype = result
        method.argtypes = arguments
    return lib


def _array(type_, values):
    values = tuple(values)
    return (type_ * len(values))(*values)


def _decode(values, length):
    return tuple(values[i].decode() for i in range(length))


def _position(position):
    return _Position(position.group, position.end_token)


def _operation(operation):
    return _Operation(operation.instance, operation.operation)


def _keys(keys):
    values = _array(_Bytes, map(_Bytes.encode, keys))
    return _Keys(values, len(values))


def _pages(pages):
    return _array(_Page, (_Page(p.pool, p.slot, p.generation, p.hold) for p in pages))


def _bindings(bindings):
    def encode(item):
        pages = _pages(item.pages)
        return _Binding(_position(item.position), pages, len(pages))

    return _array(_Binding, map(encode, bindings))


class Plan:
    def __init__(self, lib, handle):
        self._lib, self._handle = lib, handle
        result = lib.btc_plan_get(handle)
        self.request = result.request
        self.ready_tokens = result.ready_tokens
        self.recoverable_tokens = result.recoverable_tokens
        self.host_tokens = result.host_tokens
        self.pages = _decode(result.pages, result.page_count)
        self.ready_pages = self.pages if self.ready_tokens == self.recoverable_tokens else _decode(
            result.ready_pages, result.ready_page_count
        )

    def close(self):
        if self._handle:
            self._lib.btc_plan_free(self._handle)
            self._handle = None

    def __del__(self):
        self.close()


class Cache:
    def __init__(self, namespace, key_tokens, groups, host_slots=0, disk_slots=0,
                 disk_directory=None, routes=None, library_path=None, *,
                 tiered_eviction=False, release_local_pages=False, device_eviction_policy=0,
                 external_disk_io=False, host_region: HostRegion | None = None):
        def group_config(group):
            pools = group.member_pools
            if pools is not None and len(pools) != group.members:
                raise ValueError("BTC member pool count does not match group members")
            return _Group(group.id, group.rule, group.page_tokens, group.window_tokens,
                          group.members, None if pools is None else _array(_U64, pools),
                          group.host_slots, group.disk_slots)

        def route_config(item):
            kind, ids = item
            participants = _array(_U32, ids)
            return _Route(kind, participants, len(participants))

        self._lib = _load_library(library_path, with_host_region=host_region is not None)
        self._handle = _Ptr()
        self._host_memory = None
        specifications = _array(_Group, map(group_config, groups))
        directions = _array(_Route, map(route_config, (routes or {}).items()))
        config = _Config(ABI_VERSION, _Bytes.encode(namespace), key_tokens, specifications,
                         len(specifications), host_slots, disk_slots,
                         _Bytes.encode(os.fsencode(disk_directory) if disk_directory else b""),
                         directions, len(directions), release_local_pages,
                         device_eviction_policy, external_disk_io, tiered_eviction)
        if host_region is None:
            self._check(self._lib.btc_open(_c.byref(config), _c.byref(self._handle)))
        else:
            # Keeping the export alive also prevents resize/unmap while a native
            # request or an unacknowledged device transfer still holds a slot.
            memory = memoryview(host_region.buffer).cast("B")
            try:
                if memory.readonly or not memory:
                    raise ValueError("Host region must export nonempty writable storage")
                if not 0 < host_region.payload_bytes <= host_region.stride_bytes <= _Size(-1).value:
                    raise ValueError("Invalid Host row payload or stride")
                base = _c.addressof(_c.c_ubyte.from_buffer(memory))
                region = _HostRegion(_Buffer(base, len(memory)), host_region.stride_bytes, host_region.payload_bytes)
                self._check(self._lib.btc_open_with_host_region(_c.byref(config), _c.byref(region),
                                                               _c.byref(self._handle)))
            except BaseException:
                memory.release()
                raise
            self._host_memory = memory
            self._host_base = base

    def _check(self, value, permitted=()):
        status = Status(value)
        if status != Status.OK and status not in permitted:
            raw = self._lib.btc_last_error()
            raise Error(status, raw.decode(errors="replace"))
        return status

    def _live(self):
        if not self._handle:
            raise Error(Status.CLOSED, "BlockTreeCache is closed")
        return self._handle

    def host_slot_buffer(self, group: int, slot: int) -> memoryview:
        """Borrow a slot's payload for the lifetime of the caller's transfer."""
        handle = self._live()
        if self._host_memory is None or not (0 <= group < 2**32 and 0 <= slot < 2**64):
            raise ValueError("Expected a mapped Host pool and a valid group and slot")
        row = _Buffer()
        self._check(self._lib.btc_host_slot_buffer(handle, group, slot, _c.byref(row)))
        start = row.data - self._host_base
        assert 0 <= start <= len(self._host_memory) - row.size, "HostBlockPool returned a row outside its mapping"
        return self._host_memory[start:start + row.size]

    def match(self, request, keys, limits):
        groups = None if limits.groups is None else _array(_U32, limits.groups)
        boundaries = None if limits.boundaries is None else _array(_U64, limits.boundaries)
        native = _Limits(limits.max_tokens, groups, len(groups) if groups is not None else 0,
                         boundaries, len(boundaries) if boundaries is not None else 0,
                         limits.max_load_pages, limits.allow_remote)
        result = _Ptr()
        args = (self._live(), request, _keys(keys), _c.byref(native))
        cap = limits.max_load_tokens
        if cap is None:
            status = self._lib.btc_match(*args, _c.byref(result))
        else:
            if type(cap) is not int or not 0 <= cap < 2**64:
                raise ValueError("max_load_tokens must fit an unsigned 64-bit integer")
            status = self._lib.btc_match_capped(*args, cap, _c.byref(result))
        self._check(status, (Status.PENDING, Status.WAITING))
        return Plan(self._lib, result) if status == Status.OK else None

    def poll_match(self, request):
        """Collect the existing admission once its local and remote pages are ready."""
        result = _Ptr()
        status = self._check(self._lib.btc_poll_match(self._live(), request, _c.byref(result)), (Status.PENDING,))
        return Plan(self._lib, result) if status == Status.OK else None

    def bind(self, plan, bindings=()):
        native = _bindings(bindings)
        return self._check(self._lib.btc_bind(self._live(), plan._handle, native, len(native)),
                           (Status.WAITING, Status.RETRY))

    def publish(self, keys, committed_tokens, bindings):
        native = _bindings(bindings)
        return self._check(self._lib.btc_publish(self._live(), _keys(keys), committed_tokens,
                                                 native, len(native)), (Status.RETRY,))

    def poll(self):
        result = _Ptr()
        self._check(self._lib.btc_poll(self._live(), _c.byref(result)))
        try:
            view = self._lib.btc_batch_get(result)
            return Batch(_decode(view.work, view.work_count), _decode(view.events, view.event_count))
        finally:
            self._lib.btc_batch_free(result)

    def pending_work(self):
        count = _Size()
        self._check(self._lib.btc_pending_work(self._live(), _c.byref(count)))
        return count.value

    def accept(self, operation, participant):
        return self._check(self._lib.btc_accept(self._live(), _operation(operation), participant))

    def complete(self, receipts):
        def encode(receipt):
            objects = _array(_RemoteObject, (
                _RemoteObject(_position(item.position), _Bytes.encode(item.descriptor))
                for item in receipt.objects
            ))
            return _Receipt(_operation(receipt.id), receipt.participant, receipt.result, receipt.safe,
                            receipt.committed, objects, len(objects))

        native = _array(_Receipt, map(encode, receipts))
        return self._check(self._lib.btc_complete(self._live(), native, len(native)))

    def ack_events(self, cursor):
        return self._check(self._lib.btc_ack_events(self._live(), cursor))

    def reclaim(self, pool, pressure, eligible):
        native = _pages(eligible)
        return self._check(self._lib.btc_reclaim(self._live(), pool, pressure, native, len(native)))

    def host_capacity(self, group):
        """Total and immediately allocatable slots in this group's physical Host pool."""
        total, free = _Size(), _Size()
        self._check(self._lib.btc_host_capacity(self._live(), group, _c.byref(total), _c.byref(free)))
        return total.value, free.value

    def installed_device_pages(self, keys, bindings):
        native = _bindings(bindings)
        installed = (_c.c_uint8 * len(native))()
        self._check(self._lib.btc_installed_device_pages(self._live(), _keys(keys), native, len(native), installed))
        return tuple(map(bool, installed))

    def device_balance_needed(self, pool, total, free):
        needed = _c.c_uint8()
        self._check(self._lib.btc_device_balance_needed(self._live(), pool, total, free, _c.byref(needed)))
        return bool(needed.value)

    def balance_device(self, pool, total, free, eligible):
        native = _pages(eligible)
        return self._check(self._lib.btc_balance_device(self._live(), pool, total, free, native, len(native)))

    def demote(self, pool, pressure, eligible, *, remove_device=True):
        native = _pages(eligible)
        return self._check(self._lib.btc_demote(self._live(), pool, pressure, native,
                                               len(native), remove_device))

    def invalidate(self, pages):
        native = _pages(pages)
        return self._check(self._lib.btc_invalidate(self._live(), native, len(native)))

    def cancel(self, request):
        return self._check(self._lib.btc_cancel(self._live(), request))

    def drain(self):
        return self._check(self._lib.btc_drain(self._live()), (Status.PENDING,))

    def close(self):
        if not self._handle:
            return Status.OK
        status = self._check(self._lib.btc_close(self._handle), (Status.PENDING,))
        if status == Status.OK:
            self._handle = None
            if self._host_memory is not None:
                self._host_memory.release()
                self._host_memory = None
        return status

    def execute_disk(self, operation, host_buffers):
        backing = []
        for buffer in host_buffers:
            view = memoryview(buffer)
            backing.append((_c.c_ubyte * view.nbytes).from_buffer(view))
        native = _array(_Buffer, (_Buffer(_c.cast(b, _Ptr), len(b)) for b in backing))
        return self._check(self._lib.btc_execute_disk(self._live(), _operation(operation),
                                                      native, len(native)), (Status.IO,))
