# SPDX-License-Identifier: Apache-2.0
"""Python bindings to the optional BTC KVCM Host backend."""

from __future__ import annotations

import asyncio
import ctypes
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import time
from urllib.parse import urlsplit


_Payload = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p))


class PaceBufferOwnershipUnknown(RuntimeError):
    """The native SDK cannot prove that a Host span is safe to reuse."""


class BtcHostOwnershipUnknown(RuntimeError):
    """A worker may still be accessing a Host slot after losing its reply."""


@lru_cache(maxsize=1)
def _library():
    path = os.environ.get("RTP_BTC_KVCM_LIBRARY")
    if path is None:
        path = Path(__file__).with_name("_btc_kvcm_lib") / "libbtc_kvcm.so"
    lib = ctypes.CDLL(str(path))
    ptr = ctypes.c_void_p
    text = ctypes.c_char_p
    functions = {
        "last_error": (text, ()),
        "free": (None, (ptr,)),
        "create_metadata": (ptr, (text,)),
        "destroy_metadata": (None, (ptr,)),
        "metadata": (ctypes.c_int, (ptr, text, ctypes.POINTER(ptr))),
        "write_group": (ctypes.c_int, (ptr, ctypes.c_int64, ctypes.c_int64, ctypes.c_int, _Payload)),
        "create_transfer": (ptr, (ptr, ctypes.c_size_t, ctypes.c_int)),
        "destroy_transfer": (None, (ptr,)),
        "transfer": (ctypes.c_int, (ptr, text, ctypes.POINTER(ptr))),
    }
    for name, (result, args) in functions.items():
        func = getattr(lib, "btc_kvcm_" + name)
        func.restype = result
        func.argtypes = args
    return lib


def _json(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode()


def _raise_native_error(lib, status):
    message = lib.btc_kvcm_last_error().decode(errors="replace")
    if status == -2:
        raise PaceBufferOwnershipUnknown(message)
    raise RuntimeError(message)


def _invoke(lib, function, handle, op, **fields):
    if not handle:
        raise RuntimeError("BTC KVCM handle is closed")
    result = ctypes.c_void_p()
    status = function(handle, _json(dict(op=op, **fields)), ctypes.byref(result))
    if status:
        _raise_native_error(lib, status)
    try:
        return json.loads(ctypes.string_at(result))
    finally:
        lib.btc_kvcm_free(result)


@dataclass(frozen=True)
class PaceSpec:
    """A location-name, byte offset and packed per-rank lengths."""

    name: str
    offset: int
    rank_bytes: tuple[int, ...]

    def __post_init__(self):
        if not (
            isinstance(self.name, str)
            and self.name
            and type(self.offset) is int
            and self.offset >= 0
            and self.rank_bytes
            and all(type(v) is int and v > 0 for v in self.rank_bytes)
        ):
            raise ValueError(f"Invalid BTC PACE spec {self}")

    @property
    def size(self):
        return sum(self.rank_bytes)


def _signed_key(key):
    return int.from_bytes(key[-8:].rjust(8, b"\0"), "big", signed=True)


def _locations(group, key, descriptor):
    stored_group, stored_key, locations = json.loads(descriptor)
    if (stored_group, stored_key) != (group, key.hex()):
        raise ValueError("KVCM read descriptor belongs to another cache page")
    return locations


def _read_current(storage, items):
    items = tuple(items)
    descriptions = storage.describe((group, key) for group, key, _ in items)
    if any((group, key) not in descriptions for group, key, _ in items):
        return False
    return storage.read_described([
        (group, key, dest, descriptions[group, key]) for group, key, dest in items
    ])


class PaceMetadata:
    """BTC-owned native KVCM registration, lookup and write sessions."""

    def __init__(self, config, specs_by_group):
        self._handle = None
        endpoint = config["kvcm_url"].rstrip("/")
        url = urlsplit(endpoint)
        rpc_address = config.get("kvcm_rpc_address", "")
        instance = config["instance_id"]
        instance_group = config["instance_group"]
        block_size = config["block_size"]
        replicas = config.get("min_replica_count", 1)
        write_timeout = config.get("write_timeout_seconds", 30)
        commit_timeout = float(config.get("commit_timeout_s", 5))
        request_timeout = float(config.get("request_timeout_s", 5))
        if url.scheme not in ("http", "https") or not url.hostname:
            raise ValueError("BTC PACE requires a KVCM checksum HTTP endpoint")
        if not isinstance(rpc_address, str):
            raise ValueError("BTC PACE metadata RPC address must be a string")
        if not all(isinstance(v, str) and v for v in (instance, instance_group)):
            raise ValueError("BTC PACE requires an instance ID and instance group")
        if any(
            type(v) is not int or v <= 0 for v in (block_size, replicas, write_timeout)
        ):
            raise ValueError(
                "BTC PACE requires positive sizes, replicas and write timeout"
            )
        if any(
            not math.isfinite(v) or v <= 0 for v in (commit_timeout, request_timeout)
        ):
            raise ValueError("BTC PACE requires finite, positive timeouts")

        self._specs = {group: tuple(specs) for group, specs in specs_by_group.items()}
        groups = [
            dict(id=str(group), specs=[dict(name=spec.name, size=spec.size, ranks=len(spec.rank_bytes)) for spec in specs])
            for group, specs in self._specs.items()
        ]

        deployment = dict(
            model_name="default",
            dtype="bytes",
            use_mla=False,
            tp_size=1,
            dp_size=1,
            pp_size=1,
            use_eagle_pop=config.get("use_eagle_pop", False),
        )
        deployment.update(config.get("model_deployment", {}))
        sdk_config = dict(
            thread_num=4,
            queue_size=1000,
            sdk_backend_configs=[],
            timeout_config={"get_timeout_ms": 5000, "put_timeout_ms": 10000},
        )
        sdk_config.update(config.get("sdk_config", {}))
        options = dict(
            instance_group=instance_group,
            instance_id=instance,
            block_size=block_size,
            sdk_config=sdk_config,
        )
        registration = dict(
            instance_group=instance_group,
            instance_id=instance,
            block_size=block_size,
            model_deployment=deployment,
            default_query_type="QT_PREFIX_MATCH_WITH_MAMBA",
        )
        native_config = dict(
            url=endpoint,
            rpc_address=rpc_address,
            request_timeout_ms=math.ceil(request_timeout * 1000),
            commit_timeout_ms=math.ceil(commit_timeout * 1000),
            write_timeout_seconds=write_timeout,
            min_replica_count=replicas,
            registration=registration,
            options=options,
            groups=groups,
        )
        self._lib = _library()
        self._handle = self._lib.btc_kvcm_create_metadata(_json(native_config))
        if not self._handle:
            raise RuntimeError(self._lib.btc_kvcm_last_error().decode(errors="replace"))

    def _call(self, op, **fields):
        return _invoke(
            self._lib, self._lib.btc_kvcm_metadata, self._handle, op, **fields
        )

    def connect(self):
        return tuple(self._call("connect"))

    def proofs(self, items):
        """Map ((group, hash), spec name) to an address and rank digests."""
        lines = [(group, _signed_key(key), spec.name) for (group, key), spec in items]
        proofs = self._call("proofs", items=lines)
        return {
            (items[index][0], items[index][1].name): (uri, digests)
            for index, uri, digests in proofs
        }

    def lookup(self, keys):
        return set(self.describe(keys))

    def describe(self, keys):
        """Capture the exact URIs and checksums found at match time."""
        keys = tuple(dict.fromkeys(keys))
        rows = [(group, _signed_key(key), key.hex()) for group, key in keys]
        descriptions = self._call("describe", rows=rows)
        return {keys[index]: descriptor.encode() for index, descriptor in descriptions}

    def write_group_via(self, block, write, *, expected_digests=None):
        group, key = block
        if not self._handle:
            raise RuntimeError("BTC KVCM handle is closed")
        encoded = error = None

        @_Payload
        def execute(data, output):
            nonlocal encoded, error
            try:
                request = json.loads(data)
                if request["op"] == "digest":
                    assert expected_digests is not None
                    result = expected_digests()
                else:
                    result = write(request["uris"])
                encoded = ctypes.create_string_buffer(_json(result))
                output[0] = ctypes.cast(encoded, ctypes.c_void_p)
                return 0
            except BaseException as exc:
                error = exc
                return -1

        status = self._lib.btc_kvcm_write_group(
            self._handle, group, _signed_key(key), expected_digests is not None, execute
        )
        if error is not None:
            raise error
        if status < 0:
            _raise_native_error(self._lib, status)
        return bool(status)

    def close(self):
        if self._handle:
            self._lib.btc_kvcm_destroy_metadata(self._handle)
            self._handle = None

    def __del__(self):
        self.close()


class PaceHostTransfer:
    """Shared KVCM SDK client borrowing one fd-backed BTC Host mapping."""

    def __init__(self, memory, memory_fd):
        self._handle = None
        self._closed = True
        if type(memory_fd) is not int or memory_fd < 0:
            raise ValueError("BTC PACE requires an open fd backing its Host mmap")
        self._memory = memoryview(memory).cast("B")
        if self._memory.readonly or self._memory.nbytes == 0:
            self._memory.release()
            raise ValueError("BTC PACE requires a writable contiguous Host mmap")
        try:
            self._base = ctypes.addressof(ctypes.c_char.from_buffer(self._memory))
            self._lib = _library()
            self._handle = self._lib.btc_kvcm_create_transfer(
                self._base,
                self._memory.nbytes,
                memory_fd,
            )
            if not self._handle:
                raise RuntimeError(
                    self._lib.btc_kvcm_last_error().decode(errors="replace")
                )
        except BaseException:
            self._memory.release()
            raise
        self._closed = False

    def _call(self, op, **fields):
        return _invoke(
            self._lib, self._lib.btc_kvcm_transfer, self._handle, op, **fields
        )

    def connect(self, storage_configs, options, self_location_spec_name):
        self._call(
            "connect",
            storage=storage_configs,
            options=options,
            name=self_location_spec_name,
        )

    def offset(self, view, size):
        row = view.cast("B")
        if row.readonly or row.nbytes < size:
            raise ValueError("BTC PACE needs a writable, complete Host row")
        offset = ctypes.addressof(ctypes.c_char.from_buffer(row)) - self._base
        if not 0 <= offset <= self._memory.nbytes - size:
            raise ValueError("BTC PACE Host span lies outside the registered mmap")
        return offset

    def read_spec(self, uri, spec, offset, digests):
        return self.read_many([(uri, spec, offset, digests)])

    def read_many(self, items):
        return self._call(
            "read",
            entries=[
                (uri, offset, spec.rank_bytes, digests)
                for uri, spec, offset, digests in items
            ],
        )

    def digest(self, spec, offset):
        return self._call("digest", rank_bytes=spec.rank_bytes, offset=offset)

    def write_spec(self, uri, spec, offset):
        return self._call("write", uri=uri, rank_bytes=spec.rank_bytes, offset=offset)

    def close(self):
        if not self._closed:
            self._lib.btc_kvcm_destroy_transfer(self._handle)
            self._handle = None
            self._memory.release()
            self._closed = True

    def __del__(self):
        self.close()


class PaceStorageBackend:
    """KVCM and PACE for an engine whose Host rows fit in one mapping."""

    def __init__(self, config, rank_bytes, ranks, *, memory, memory_fd):
        if any(type(v) is not int or v <= 0 for v in (rank_bytes, ranks)):
            raise ValueError("BTC PACE requires positive rank count and rank size")
        self._payload = rank_bytes * ranks
        self._specs = {group: tuple(specs) for group, specs in config["specs"].items()}
        for specs in self._specs.values():
            end = 0
            if not specs:
                raise ValueError("BTC PACE requires at least one spec per cache group")
            for spec in sorted(specs, key=lambda spec: spec.offset):
                if spec.offset < end or spec.offset + spec.size > self._payload:
                    raise ValueError(f"Invalid BTC PACE Host slice {spec}")
                end = spec.offset + spec.size
        if memory.nbytes < self._payload:
            raise ValueError("BTC PACE requires one complete Host row")
        deployment = {"tp_size": ranks, **config.get("model_deployment", {})}
        self._metadata = PaceMetadata(
            dict(config, model_deployment=deployment), self._specs
        )
        self._self_spec = next(iter(self._specs.values()))[0].name
        self._host = PaceHostTransfer(memory, memory_fd)
        self._connected = False

    def _connect(self):
        if not self._connected:
            storage, options = self._metadata.connect()
            self._host.connect(storage, options, self._self_spec)
            self._connected = True

    def lookup(self, keys):
        return self._metadata.lookup(keys)

    def describe(self, keys):
        return self._metadata.describe(keys)

    def read(self, group, key, dest):
        return self.read_many([(group, key, dest)])

    def read_many(self, items):
        return _read_current(self, items)

    def read_described(self, items):
        self._connect()
        reads = []
        for group, key, dest, descriptor in items:
            locations = _locations(group, key, descriptor)
            offset = self._host.offset(dest, self._payload)
            for spec in self._specs[group]:
                uri, digests = locations[spec.name]
                reads.append((uri, spec, offset + spec.offset, digests))
        return self._host.read_many(reads)

    def write(self, group, key, source):
        offset = self._host.offset(source, self._payload)
        self._connect()
        specs = self._specs[group]

        def save(uris):
            return {
                spec.name: self._host.write_spec(
                    uris[spec.name], spec, offset + spec.offset
                )
                for spec in specs
                if spec.name in uris
            }

        def digest():
            return {
                spec.name: self._host.digest(spec, offset + spec.offset)
                for spec in specs
            }

        return self._metadata.write_group_via(
            (group, key), save, expected_digests=digest
        )

    def close(self):
        try:
            self._host.close()
        finally:
            self._metadata.close()


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
        if not math.isfinite(self._timeout) or self._timeout <= 0:
            raise ValueError("BTC PACE worker_timeout_s must be finite and positive")
        if (
            type(write_timeout) is not int
            or type(sdk_put_ms) is not int
            or not math.isfinite(request_timeout)
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

    def describe(self, keys):
        return self._metadata.describe(keys)

    def read_many(self, items):
        return _read_current(self, items)

    def read_described(self, items):
        self._connect()
        by_rank = defaultdict(list)
        for group, key, slot, descriptor in items:
            locations = _locations(group, key, descriptor)
            for rank, spec in self._group_specs[group].items():
                uri, digests = locations[spec.name]
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
