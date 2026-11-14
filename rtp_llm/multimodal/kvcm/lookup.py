"""Read-only KVMeta Get, without exposing unpinned storage URIs to callers."""

import threading
import time
import uuid

import grpc

from . import lookup_pb2
from ._config import _resolve_vipserver_addresses


def _cached_vipserver_resolver(domain):
    # Config initialization has already seeded RTP's existing refresh machinery.
    from rtp_llm.vipserver import get_host_list_by_domain

    return get_host_list_by_domain(domain)


class KvMetaLookupError(RuntimeError):
    def __init__(self, message, *, reason="metadata", code=None):
        super().__init__(message)
        self.reason = reason
        self.code = code


class KvMetaLookup:
    def __init__(self, config):
        self._config = config
        self._lock = threading.Lock()
        self._closed = False
        self._addresses = tuple(config.addresses)
        self._channels = []
        self._set_endpoints(self._addresses)

    def _set_endpoints(self, addresses):
        old_channels = self._channels
        self._channels = [grpc.insecure_channel(address) for address in addresses]
        self._calls = [
            channel.unary_unary(
                "/kv_cache_manager.proto.kv_meta.MetaService/Get",
                request_serializer=lookup_pb2.GetRequest.SerializeToString,
                response_deserializer=lookup_pb2.GetResponse.FromString,
            )
            for channel in self._channels
        ]
        self._addresses = tuple(addresses)
        # A discovery change may cancel an old metadata read; callers fall back
        # without starting a payload transfer. Stale channels are never retained.
        for channel in old_channels:
            channel.close()

    def size(self, key, *, trace_id=None, timeout_ms=None):
        if not isinstance(key, str) or not key or len(key.encode()) > 512:
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
            domain = getattr(self._config, "vipserver_domain", "")
            if domain:
                try:
                    addresses = _resolve_vipserver_addresses(
                        domain, _cached_vipserver_resolver
                    )
                except Exception:
                    raise KvMetaLookupError(
                        "KVMeta service discovery unavailable", reason="discovery"
                    ) from None
                if set(addresses) != set(self._addresses):
                    self._set_endpoints(addresses)
            calls = list(self._calls)
        last_code, last_reason = None, "deadline"
        for index, call in enumerate(calls):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                response = call(request, timeout=remaining / (len(calls) - index))
            except grpc.RpcError as error:
                last_reason = "grpc"
                try:
                    last_code = error.code()
                except Exception:
                    last_code = None
                continue
            status = response.header.status.code
            if status in (4, 9):  # Not ready / not leader: read-only failover.
                last_code, last_reason = status, "service_status"
                continue
            if status != 1:
                raise KvMetaLookupError(
                    f"KVMeta Get failed with status {status}",
                    reason="service_status",
                    code=status,
                )
            if len(response.locations) != 1 or len(response.hit_mask.values) != 1:
                raise KvMetaLookupError(
                    "KVMeta Get returned a mismatched result",
                    reason="response_mismatch",
                )
            if not response.hit_mask.values[0]:
                return None
            size = response.locations[0].value_size
            if not 0 < size <= self._config.max_object_bytes:
                raise KvMetaLookupError(
                    "KVMeta object exceeds configured byte limit", reason="object_size"
                )
            return size
        raise KvMetaLookupError(
            "KVMeta Get unavailable within deadline", reason=last_reason, code=last_code
        )

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            channels, self._channels = self._channels, []
        for channel in channels:
            channel.close()
