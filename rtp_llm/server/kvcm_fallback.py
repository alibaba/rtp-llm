"""Lightweight KVCM cache-affinity fallback for :mod:`master_client`.

The client deliberately does not recreate FlexLB load accounting.  It queries
KVCM only after a FlexLB availability failure and selects the worker with the
largest positive local prefix match.  Callers retain their existing final
domain/upstream fallback when KVCM cannot provide a route.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Optional

import grpc

from rtp_llm.server.kvcm_proto import kvcm_meta_service_pb2 as kvcm_pb2
from rtp_llm.server.kvcm_proto import kvcm_meta_service_pb2_grpc as kvcm_pb2_grpc


class KvcmFallbackError(RuntimeError):
    """KVCM could not produce a trustworthy cache route."""


@dataclass(frozen=True)
class KvcmFallbackConfig:
    """Static metadata required to query one KVCM cache namespace."""

    instance_id: str
    block_size: int
    request_timeout_ms: int = 100
    leader_refresh_interval_ms: int = 10_000
    p2p_host_count: int = 0
    worker_grpc_port_override: Optional[int] = None
    lookahead_tokens: int = 0
    minimum_local_blocks: int = 1

    def __post_init__(self) -> None:
        if not self.instance_id.strip():
            raise ValueError("KVCM instance_id must not be empty")
        if self.block_size <= 0:
            raise ValueError("KVCM block_size must be positive")
        if self.request_timeout_ms <= 0:
            raise ValueError("KVCM request_timeout_ms must be positive")
        if self.leader_refresh_interval_ms <= 0:
            raise ValueError("KVCM leader_refresh_interval_ms must be positive")
        if self.p2p_host_count < 0:
            raise ValueError("KVCM p2p_host_count must not be negative")
        if self.lookahead_tokens < 0:
            raise ValueError("KVCM lookahead_tokens must not be negative")
        if self.minimum_local_blocks <= 0:
            raise ValueError("KVCM minimum_local_blocks must be positive")
        if self.worker_grpc_port_override is not None and not (
            1 <= self.worker_grpc_port_override <= 65_535
        ):
            raise ValueError("KVCM worker_grpc_port_override must be a valid port")


@dataclass(frozen=True)
class KvcmCacheCandidate:
    host_ip: str
    http_port: int
    grpc_port: int
    local_blocks: int
    p2p_fetch_blocks: int
    p2p_total_match_blocks: int

    @property
    def host_ip_port(self) -> str:
        address = ipaddress.ip_address(self.host_ip)
        if address.version == 6:
            return f"[{address.compressed}]:{self.http_port}"
        return f"{address.compressed}:{self.http_port}"


@dataclass(frozen=True)
class KvcmFallbackResult:
    outcome: str
    selected: Optional[KvcmCacheCandidate]
    candidate_count: int
    block_count: int
    latency_us: int


BootstrapResolver = Callable[[], Sequence[str]]


def _cbor_head(major_type: int, value: int) -> bytes:
    if value < 0:
        raise ValueError("CBOR length/value must not be negative")
    prefix = major_type << 5
    if value < 24:
        return bytes((prefix | value,))
    if value <= 0xFF:
        return bytes((prefix | 24, value))
    if value <= 0xFFFF:
        return bytes((prefix | 25,)) + value.to_bytes(2, "big")
    if value <= 0xFFFFFFFF:
        return bytes((prefix | 26,)) + value.to_bytes(4, "big")
    if value <= 0xFFFFFFFFFFFFFFFF:
        return bytes((prefix | 27,)) + value.to_bytes(8, "big")
    raise ValueError("CBOR integer is outside uint64")


def _cbor_int(value: int) -> bytes:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("token ids must be integers")
    return _cbor_head(0, value) if value >= 0 else _cbor_head(1, -1 - value)


def _encode_vllm_hash_input(parent_hash: bytes, token_ids: Sequence[int]) -> bytes:
    encoded = bytearray()
    encoded.extend(_cbor_head(4, 3))
    encoded.extend(_cbor_head(2, len(parent_hash)))
    encoded.extend(parent_hash)
    encoded.extend(_cbor_head(4, len(token_ids)))
    for token_id in token_ids:
        encoded.extend(_cbor_int(token_id))
    encoded.append(0xF6)  # CBOR null: vLLM's extra-hash field.
    return bytes(encoded)


def calculate_vllm_block_cache_keys(
    input_ids: Sequence[int],
    block_size: int,
    lookahead_tokens: int = 0,
) -> list[int]:
    """Return vLLM ``sha256_cbor`` low-64 block keys.

    This matches vLLM with ``prefix_caching_hash_algo=sha256_cbor`` and
    ``PYTHONHASHSEED=0``.  Only complete stride blocks are returned; lookahead
    tokens participate in each hash but do not change the stride.
    """

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if lookahead_tokens < 0:
        raise ValueError("lookahead_tokens must not be negative")
    if not input_ids or len(input_ids) < block_size:
        return []

    # Canonical CBOR text string "0" is 0x61 0x30.
    parent_hash = hashlib.sha256(b"\x61\x30").digest()
    keys: list[int] = []
    full_block_count = len(input_ids) // block_size
    for block_index in range(full_block_count):
        offset = block_index * block_size
        remaining_after_block = len(input_ids) - offset - block_size
        token_count = block_size + min(lookahead_tokens, remaining_after_block)
        block_tokens = input_ids[offset : offset + token_count]
        parent_hash = hashlib.sha256(
            _encode_vllm_hash_input(parent_hash, block_tokens)
        ).digest()
        keys.append(int.from_bytes(parent_hash[-8:], "big", signed=True))
    return keys


def select_max_local_affinity(
    candidates: Sequence[KvcmCacheCandidate],
    minimum_local_blocks: int = 1,
) -> Optional[KvcmCacheCandidate]:
    """Select the largest positive local match with a deterministic tie break."""

    eligible = [
        candidate
        for candidate in candidates
        if candidate.local_blocks >= minimum_local_blocks
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda candidate: (-candidate.local_blocks, candidate.host_ip_port),
    )


def _split_ip_port(authority: str) -> tuple[str, int]:
    authority = authority.strip()
    if authority.startswith("["):
        closing = authority.find("]")
        if closing <= 1 or authority[closing + 1 : closing + 2] != ":":
            raise ValueError("invalid bracketed host endpoint")
        host = authority[1:closing]
        port_text = authority[closing + 2 :]
    else:
        if authority.count(":") != 1:
            raise ValueError("host endpoint must be an IP:port authority")
        host, port_text = authority.rsplit(":", 1)
    address = ipaddress.ip_address(host)
    port = int(port_text, 10)
    if not 1 <= port <= 65_535:
        raise ValueError("host endpoint port is invalid")
    return address.compressed, port


def _format_target(host: str, port: int) -> str:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return f"{host}:{port}"
    return (
        f"[{address.compressed}]:{port}"
        if address.version == 6
        else f"{address.compressed}:{port}"
    )


class KvcmFallbackClient:
    """Async KVCM client with lazy leader discovery and reusable channels."""

    def __init__(
        self,
        config: KvcmFallbackConfig,
        bootstrap_resolver: BootstrapResolver,
    ) -> None:
        self.config = config
        self._bootstrap_resolver = bootstrap_resolver
        self._channels: dict[str, grpc.aio.Channel] = {}
        self._stubs: dict[str, kvcm_pb2_grpc.MetaServiceStub] = {}
        self._leader: Optional[str] = None
        self._leader_refreshed_at = 0.0
        self._leader_lock = asyncio.Lock()
        self._closed = False

    def _stub_for(self, target: str) -> kvcm_pb2_grpc.MetaServiceStub:
        if self._closed:
            raise KvcmFallbackError("KVCM fallback client is closed")
        stub = self._stubs.get(target)
        if stub is not None:
            return stub
        channel = grpc.aio.insecure_channel(
            target,
            options=(
                ("grpc.keepalive_time_ms", 30_000),
                ("grpc.keepalive_timeout_ms", 5_000),
            ),
        )
        stub = kvcm_pb2_grpc.MetaServiceStub(channel)
        self._channels[target] = channel
        self._stubs[target] = stub
        return stub

    async def _resolve_leader(self, trace_id: str) -> str:
        now = time.monotonic()
        refresh_interval_s = self.config.leader_refresh_interval_ms / 1_000.0
        if self._leader and now - self._leader_refreshed_at < refresh_interval_s:
            return self._leader

        async with self._leader_lock:
            now = time.monotonic()
            if self._leader and now - self._leader_refreshed_at < refresh_interval_s:
                return self._leader
            previous_leader = self._leader
            targets = tuple(dict.fromkeys(self._bootstrap_resolver()))
            timeout_s = self.config.request_timeout_ms / 1_000.0
            for target in targets:
                try:
                    response = await self._stub_for(target).GetClusterInfo(
                        kvcm_pb2.GetClusterInfoRequest(trace_id=trace_id),
                        timeout=timeout_s,
                    )
                except (grpc.RpcError, asyncio.TimeoutError):
                    continue
                if response.header.status.code != kvcm_pb2.OK:
                    continue
                if not response.HasField("leader_endpoint"):
                    continue
                endpoint = response.leader_endpoint
                if not endpoint.host or not 1 <= endpoint.meta_rpc_port <= 65_535:
                    continue
                self._leader = _format_target(endpoint.host, endpoint.meta_rpc_port)
                self._leader_refreshed_at = time.monotonic()
                return self._leader

            # A failed refresh should not discard a previously working target;
            # the following query will cheaply prove whether it is still usable.
            if previous_leader:
                self._leader_refreshed_at = time.monotonic()
                return previous_leader
            raise KvcmFallbackError("KVCM leader is unavailable")

    def _invalidate_leader(self, target: str) -> None:
        if self._leader == target:
            self._leader = None
            self._leader_refreshed_at = 0.0

    def _parse_candidates(
        self, hosts: Sequence[kvcm_pb2.HostCacheMatch]
    ) -> list[KvcmCacheCandidate]:
        candidates: list[KvcmCacheCandidate] = []
        for host in hosts:
            try:
                host_ip, http_port = _split_ip_port(host.host_ip_port)
            except (TypeError, ValueError):
                continue
            grpc_port = self.config.worker_grpc_port_override or http_port + 1
            if grpc_port > 65_535:
                continue
            candidates.append(
                KvcmCacheCandidate(
                    host_ip=host_ip,
                    http_port=http_port,
                    grpc_port=grpc_port,
                    local_blocks=max(0, int(host.local)),
                    p2p_fetch_blocks=max(0, int(host.p2p_1_fetch)),
                    p2p_total_match_blocks=max(0, int(host.p2p_1_total_match)),
                )
            )
        return candidates

    async def query_and_select(
        self,
        *,
        request_id: str,
        block_cache_keys: Sequence[int],
        input_ids: Optional[Sequence[int]] = None,
    ) -> KvcmFallbackResult:
        """Query KVCM and return the maximum-local-affinity candidate."""

        started_at = time.monotonic_ns()
        keys = list(block_cache_keys)
        if not keys and input_ids is not None:
            keys = calculate_vllm_block_cache_keys(
                input_ids,
                self.config.block_size,
                self.config.lookahead_tokens,
            )
        if not keys:
            return KvcmFallbackResult(
                outcome="no_complete_blocks",
                selected=None,
                candidate_count=0,
                block_count=0,
                latency_us=max(0, (time.monotonic_ns() - started_at) // 1_000),
            )

        leader = await self._resolve_leader(request_id)
        request = kvcm_pb2.GetHostCacheStateRequest(
            trace_id=request_id,
            instance_id=self.config.instance_id,
            query_type=kvcm_pb2.QT_PREFIX_MATCH,
            block_cache_keys=keys,
            p2p_host_count=self.config.p2p_host_count,
        )
        try:
            response = await self._stub_for(leader).GetHostCacheState(
                request,
                timeout=self.config.request_timeout_ms / 1_000.0,
            )
        except (grpc.RpcError, asyncio.TimeoutError) as error:
            self._invalidate_leader(leader)
            raise KvcmFallbackError("KVCM GetHostCacheState RPC failed") from error

        code = response.header.status.code
        if code != kvcm_pb2.OK:
            if code == kvcm_pb2.SERVER_NOT_LEADER:
                self._invalidate_leader(leader)
            raise KvcmFallbackError(
                "KVCM GetHostCacheState failed: "
                f"code={kvcm_pb2.ErrorCode.Name(code)}, "
                f"message={response.header.status.message}"
            )

        candidates = self._parse_candidates(response.hosts)
        selected = select_max_local_affinity(
            candidates,
            self.config.minimum_local_blocks,
        )
        return KvcmFallbackResult(
            outcome="selected" if selected is not None else "no_positive_match",
            selected=selected,
            candidate_count=len(candidates),
            block_count=len(keys),
            latency_us=max(0, (time.monotonic_ns() - started_at) // 1_000),
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        channels = list(self._channels.values())
        self._channels.clear()
        self._stubs.clear()
        for channel in channels:
            await channel.close()


__all__ = [
    "KvcmCacheCandidate",
    "KvcmFallbackClient",
    "KvcmFallbackConfig",
    "KvcmFallbackError",
    "KvcmFallbackResult",
    "calculate_vllm_block_cache_keys",
    "select_max_local_affinity",
]
