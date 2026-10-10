"""Read-only KVMeta Get, without exposing unpinned storage URIs to callers."""

import threading
import time
import uuid

import grpc

from . import lookup_pb2


class KvMetaLookupError(RuntimeError):
    pass


class KvMetaLookup:
    def __init__(self, config):
        self._config = config
        self._lock = threading.Lock()
        self._closed = False
        self._channels = [
            grpc.insecure_channel(address) for address in config.addresses
        ]
        self._calls = [
            channel.unary_unary(
                "/kv_cache_manager.proto.kv_meta.MetaService/Get",
                request_serializer=lookup_pb2.GetRequest.SerializeToString,
                response_deserializer=lookup_pb2.GetResponse.FromString,
            )
            for channel in self._channels
        ]

    def size(self, key, *, trace_id=None, timeout_ms=None):
        if not isinstance(key, str) or not key or len(key.encode()) > 4096:
            raise ValueError("invalid KVMeta object key")
        timeout_ms = self._config.call_timeout_ms if timeout_ms is None else timeout_ms
        if timeout_ms <= 0:
            raise TimeoutError("KVMeta lookup deadline exceeded")
        request = lookup_pb2.GetRequest(
            instance_id=self._config.instance_id,
            trace_id=trace_id or uuid.uuid4().hex,
            query_type=1,
            keys=[key],
        )
        deadline = time.monotonic() + timeout_ms / 1000
        with self._lock:
            if self._closed:
                raise RuntimeError("KVMeta lookup is closed")
            calls = list(self._calls)
        for index, call in enumerate(calls):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                response = call(request, timeout=remaining / (len(calls) - index))
            except grpc.RpcError:
                continue
            status = response.header.status.code
            if status in (4, 9):  # Not ready / not leader: read-only failover.
                continue
            if status != 1:
                raise KvMetaLookupError(f"KVMeta Get failed with status {status}")
            if len(response.locations) != 1 or len(response.hit_mask.values) != 1:
                raise KvMetaLookupError("KVMeta Get returned a mismatched result")
            if not response.hit_mask.values[0]:
                return None
            size = response.locations[0].value_size
            if not 0 < size <= self._config.max_object_bytes:
                raise KvMetaLookupError("KVMeta object exceeds configured byte limit")
            return size
        raise KvMetaLookupError("KVMeta Get unavailable within deadline")

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            channels, self._channels = self._channels, []
        for channel in channels:
            channel.close()
