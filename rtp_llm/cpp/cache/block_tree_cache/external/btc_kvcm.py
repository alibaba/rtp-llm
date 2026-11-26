# SPDX-License-Identifier: Apache-2.0
"""Python bindings to the optional BTC KVCM Host backend."""

from __future__ import annotations

import ctypes
import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit


class PaceBufferOwnershipUnknown(RuntimeError):
    """The native SDK cannot prove that a Host span is safe to reuse."""


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


def _invoke(lib, function, handle, op, **fields):
    if not handle:
        raise RuntimeError("BTC KVCM handle is closed")
    result = ctypes.c_void_p()
    status = function(handle, _json(dict(op=op, **fields)), ctypes.byref(result))
    if status:
        message = lib.btc_kvcm_last_error().decode(errors="replace")
        if status == -2:
            raise PaceBufferOwnershipUnknown(message)
        raise RuntimeError(message)
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


class PaceMetadata:
    """BTC-owned native KVCM registration, lookup and write sessions."""

    def __init__(self, config, specs_by_group):
        self._handle = None
        endpoint = config["kvcm_url"].rstrip("/")
        url = urlsplit(endpoint)
        instance = config["instance_id"]
        instance_group = config["instance_group"]
        block_size = config["block_size"]
        replicas = config.get("min_replica_count", 1)
        write_timeout = config.get("write_timeout_seconds", 30)
        commit_timeout = float(config.get("commit_timeout_s", 5))
        request_timeout = float(config.get("request_timeout_s", 5))
        if url.scheme not in ("http", "https") or not url.hostname:
            raise ValueError("BTC PACE requires a KVCM HTTP endpoint")
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

        sizes = {}
        groups = {}
        group_names = {}
        self._specs = {group: tuple(specs) for group, specs in specs_by_group.items()}
        for group, specs in self._specs.items():
            if not specs:
                raise ValueError("BTC PACE needs at least one spec per group")
            name = f"BTC_{group}" if len(specs) > 1 else specs[0].name
            if name in group_names:
                raise ValueError(f"Duplicate BTC PACE group name {name}")
            group_names[name] = [spec.name for spec in specs]
            groups[group] = {"name": name, "specs": {}}
            for spec in specs:
                if spec.name in sizes:
                    raise ValueError(f"Duplicate BTC PACE spec {spec.name}")
                sizes[spec.name] = spec.size
                groups[group]["specs"][spec.name] = len(spec.rank_bytes)
        if not sizes:
            raise ValueError("BTC PACE needs at least one location spec")

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
            location_spec_infos=sizes,
        )
        registration = dict(
            instance_group=instance_group,
            instance_id=instance,
            block_size=block_size,
            model_deployment=deployment,
            location_spec_infos=[
                {"name": name, "size": size} for name, size in sizes.items()
            ],
            location_spec_groups=[
                {"name": name, "spec_names": specs}
                for name, specs in group_names.items()
            ],
            default_query_type="QT_PREFIX_MATCH_WITH_MAMBA",
        )
        native_config = dict(
            url=endpoint,
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
        keys = tuple(dict.fromkeys(keys))
        proofs = self.proofs(
            [(block, spec) for block in keys for spec in self._specs[block[0]]]
        )
        return {
            block
            for block in keys
            if all((block, spec.name) in proofs for spec in self._specs[block[0]])
        }

    def write_group_via(self, block, write, *, expected_digests=None):
        group, key = block
        identity = dict(group=group, key=_signed_key(key))
        plan = self._call("prepare_write", **identity)
        if "proofs" in plan:
            return expected_digests is None or plan["proofs"] == expected_digests()
        if not plan["session"]:
            expected = expected_digests() if expected_digests is not None else {}
            return self._call("confirm", **identity, expected=expected)
        # Exceptions may mean a worker still accesses the advertised URI.
        # Let the server's lease expire if that completion cannot be proved.
        written = write(plan["uris"])
        return self._call("finish_write", **identity, plan=plan, written=written)

    def close(self):
        if self._handle:
            self._lib.btc_kvcm_destroy_metadata(self._handle)
            self._handle = None

    def __del__(self):
        self.close()


class PaceHostTransfer:
    """TransferClient borrowing one fd-backed BTC Host mapping."""

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

    def read(self, group, key, dest):
        return self.read_many([(group, key, dest)])

    def read_many(self, items):
        keys = tuple(dict.fromkeys((g, key) for g, key, _ in items))
        specs = [(block, spec) for block in keys for spec in self._specs[block[0]]]
        proofs = self._metadata.proofs(specs)
        if len(proofs) != len(specs):
            return False
        self._connect()
        reads = []
        for group, key, dest in items:
            offset = self._host.offset(dest, self._payload)
            for spec in self._specs[group]:
                uri, digests = proofs[(group, key), spec.name]
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
