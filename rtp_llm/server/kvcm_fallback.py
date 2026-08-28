"""Lightweight service-discovery + KVCM fallback for MasterClient.

The implementation reuses FlexLB CacheAffinityFirst semantics over a bounded
worker subset. KVCM contributes cache-hot workers while a background VIP
snapshot contributes cache-cold workers. WorkerStatus responses are fresh for
every request; only gRPC channels and deterministic selection history are kept.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

import grpc

from rtp_llm.server.kvcm_proto import kvcm_meta_service_pb2 as kvcm_pb2
from rtp_llm.server.kvcm_proto import kvcm_meta_service_pb2_grpc as kvcm_pb2_grpc
from rtp_llm.server.worker_status_proto import worker_status_service_pb2 as status_pb2
from rtp_llm.server.worker_status_proto import (
    worker_status_service_pb2_grpc as status_pb2_grpc,
)

route_logger = logging.getLogger("route_logger")


class KvcmFallbackError(RuntimeError):
    """KVCM could not produce a trustworthy fallback decision."""


@dataclass(frozen=True)
class KvcmFallbackConfig:
    """Metadata, bounded probe settings, and FlexLB-compatible thresholds."""

    instance_id: str
    block_size: int
    request_timeout_ms: int = 100
    leader_refresh_interval_ms: int = 10_000
    p2p_host_count: int = 0
    worker_grpc_port_override: Optional[int] = None
    worker_status_port_override: Optional[int] = None
    block_hash_lookahead_tokens: int = 0
    minimum_local_blocks: int = 1
    candidate_pool_size: int = 3
    hot_candidate_pool_size: int = 2
    cold_candidate_batch_size: int = 3
    worker_status_concurrency: int = 3
    worker_status_timeout_ms: int = 200
    prefill_queue_size_threshold: int = 1_024
    p2p_hit_discount: float = 0.2
    cache_affinity_first_max_extra_work_tokens: int = 0
    outstanding_uncached_tokens_threshold: int = 0
    cache_affinity_first_min_hit_rate: float = 5.0

    def __post_init__(self) -> None:
        if not self.instance_id.strip():
            raise ValueError("KVCM instance_id must not be empty")
        for name, value in (
            ("block_size", self.block_size),
            ("request_timeout_ms", self.request_timeout_ms),
            ("leader_refresh_interval_ms", self.leader_refresh_interval_ms),
            ("candidate_pool_size", self.candidate_pool_size),
            ("hot_candidate_pool_size", self.hot_candidate_pool_size),
            ("cold_candidate_batch_size", self.cold_candidate_batch_size),
            ("worker_status_concurrency", self.worker_status_concurrency),
            ("worker_status_timeout_ms", self.worker_status_timeout_ms),
            ("prefill_queue_size_threshold", self.prefill_queue_size_threshold),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"KVCM {name} must be positive")
        if self.hot_candidate_pool_size > self.candidate_pool_size:
            raise ValueError(
                "KVCM hot_candidate_pool_size must not exceed candidate_pool_size"
            )
        for name, value in (
            ("p2p_host_count", self.p2p_host_count),
            (
                "block_hash_lookahead_tokens",
                self.block_hash_lookahead_tokens,
            ),
            (
                "cache_affinity_first_max_extra_work_tokens",
                self.cache_affinity_first_max_extra_work_tokens,
            ),
            (
                "outstanding_uncached_tokens_threshold",
                self.outstanding_uncached_tokens_threshold,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"KVCM {name} must not be negative")
        if self.minimum_local_blocks <= 0:
            raise ValueError("KVCM minimum_local_blocks must be positive")
        for name, value in (
            ("worker_grpc_port_override", self.worker_grpc_port_override),
            ("worker_status_port_override", self.worker_status_port_override),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= 65_535
            ):
                raise ValueError(f"KVCM {name} must be a valid port")
        for name, value in (
            ("p2p_hit_discount", self.p2p_hit_discount),
            (
                "cache_affinity_first_min_hit_rate",
                self.cache_affinity_first_min_hit_rate,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value < 0
            ):
                raise ValueError(f"KVCM {name} must not be negative")


@dataclass(frozen=True)
class KvcmCacheCandidate:
    host_ip: str
    http_port: int
    grpc_port: int
    worker_status_port: int
    local_blocks: int
    p2p_fetch_blocks: int
    p2p_total_match_blocks: int

    @property
    def host_ip_port(self) -> str:
        return _format_target(self.host_ip, self.http_port)

    @property
    def route_target(self) -> str:
        return _format_target(self.host_ip, self.grpc_port)

    @property
    def worker_status_target(self) -> str:
        return _format_target(self.host_ip, self.worker_status_port)


@dataclass(frozen=True)
class WorkerLoadSnapshot:
    candidate: KvcmCacheCandidate
    role: str
    alive: bool
    waiting_task_count: int
    outstanding_uncached_tokens: int
    status_version: int


@dataclass(frozen=True)
class ScoredCandidate:
    snapshot: WorkerLoadSnapshot
    hit_cache_tokens: int
    request_uncached_tokens: int
    estimated_ttft_work: int

    @property
    def candidate(self) -> KvcmCacheCandidate:
        return self.snapshot.candidate


@dataclass(frozen=True)
class CacheAffinityDecision:
    selected: Optional[ScoredCandidate]
    reason: str
    available_count: int
    eligible_count: int


@dataclass(frozen=True)
class KvcmFallbackResult:
    outcome: str
    selected: Optional[KvcmCacheCandidate]
    candidate_count: int
    block_count: int
    latency_us: int
    cache_query_outcome: str = ""
    pool_candidate_count: int = 0
    status_success_count: int = 0
    status_latency_us: int = 0
    selection_reason: Optional[str] = None
    selected_hit_cache_tokens: int = 0
    selected_outstanding_uncached_tokens: int = 0
    selected_request_uncached_tokens: int = 0
    selected_estimated_ttft_work: int = 0
    discovered_candidate_count: int = 0
    probe_round_count: int = 0


BootstrapResolver = Callable[[], Sequence[str]]
CandidateSnapshotResolver = Callable[[], Sequence[KvcmCacheCandidate]]


@dataclass(frozen=True)
class CandidatePlan:
    """One request's bounded hot set and deterministic cold ordering."""

    hot: tuple[KvcmCacheCandidate, ...]
    cold: tuple[KvcmCacheCandidate, ...]

    @property
    def candidates(self) -> tuple[KvcmCacheCandidate, ...]:
        return self.hot + self.cold

    def batches(self, cold_batch_size: int) -> list[tuple[KvcmCacheCandidate, ...]]:
        batches: list[tuple[KvcmCacheCandidate, ...]] = []
        first_cold = self.cold[:cold_batch_size]
        first = self.hot + first_cold
        if first:
            batches.append(first)
        for offset in range(cold_batch_size, len(self.cold), cold_batch_size):
            batches.append(self.cold[offset : offset + cold_batch_size])
        return batches


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
    encoded.append(0xF6)
    return bytes(encoded)


def calculate_vllm_block_cache_keys(
    input_ids: Sequence[int],
    block_size: int,
    lookahead_tokens: int = 0,
) -> list[int]:
    """Return vLLM sha256_cbor low-64 keys for complete stride blocks."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if lookahead_tokens < 0:
        raise ValueError("lookahead_tokens must not be negative")
    if not input_ids or len(input_ids) < block_size:
        return []

    parent_hash = hashlib.sha256(b"\x61\x30").digest()
    keys: list[int] = []
    full_block_count = len(input_ids) // block_size
    for block_index in range(full_block_count):
        offset = block_index * block_size
        remaining_after_block = len(input_ids) - offset - block_size
        token_count = block_size + min(lookahead_tokens, remaining_after_block)
        parent_hash = hashlib.sha256(
            _encode_vllm_hash_input(
                parent_hash,
                input_ids[offset : offset + token_count],
            )
        ).digest()
        keys.append(int.from_bytes(parent_hash[-8:], "big", signed=True))
    return keys


def select_max_local_affinity(
    candidates: Sequence[KvcmCacheCandidate],
    minimum_local_blocks: int = 1,
) -> Optional[KvcmCacheCandidate]:
    """Compatibility helper retained for raw KVCM affinity tests."""

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


def effective_cache_blocks(
    candidate: KvcmCacheCandidate,
    p2p_hit_discount: float,
) -> float:
    p2p_added = max(
        0,
        candidate.p2p_total_match_blocks - candidate.local_blocks,
    )
    return candidate.local_blocks + p2p_added * max(0.0, p2p_hit_discount)


def _positive_hot_candidates(
    candidates: Sequence[KvcmCacheCandidate],
    config: KvcmFallbackConfig,
) -> list[KvcmCacheCandidate]:
    positive = [
        candidate
        for candidate in candidates
        if candidate.local_blocks >= config.minimum_local_blocks
        or candidate.p2p_total_match_blocks >= config.minimum_local_blocks
    ]
    positive.sort(
        key=lambda candidate: (
            -effective_cache_blocks(candidate, config.p2p_hit_discount),
            -candidate.local_blocks,
            candidate.route_target,
        )
    )
    # Deduplicate before applying H. Multiple cache records for one route must
    # not shrink the effective hot pool.
    unique: list[KvcmCacheCandidate] = []
    seen: set[str] = set()
    for candidate in positive:
        if candidate.route_target in seen:
            continue
        seen.add(candidate.route_target)
        unique.append(candidate)
        if len(unique) >= config.hot_candidate_pool_size:
            break
    return unique


def _cold_rank(
    request_id: str,
    instance_id: str,
    candidate: KvcmCacheCandidate,
) -> tuple[bytes, str]:
    material = (
        request_id.encode("utf-8", errors="surrogatepass")
        + b"\x00"
        + instance_id.encode("utf-8", errors="surrogatepass")
        + b"\x00"
        + candidate.route_target.encode("utf-8", errors="surrogatepass")
    )
    return hashlib.sha256(material).digest(), candidate.route_target


def build_candidate_plan(
    candidates: Sequence[KvcmCacheCandidate],
    discovered_candidates: Sequence[KvcmCacheCandidate],
    config: KvcmFallbackConfig,
    *,
    request_id: str,
    local_candidate: Optional[KvcmCacheCandidate] = None,
) -> CandidatePlan:
    """Combine Top-H KVCM workers with a deterministic cold VIP subset.

    ``local_candidate`` remains as a compatibility input, but is treated exactly
    like any other cold discovered worker. It is never pinned to the first round.
    Candidate identity is ``(IP, route gRPC port)`` via ``route_target``.
    """

    hot = _positive_hot_candidates(candidates, config)
    hot_targets = {candidate.route_target for candidate in hot}
    cold_by_target: dict[str, KvcmCacheCandidate] = {}
    for candidate in discovered_candidates:
        if candidate.route_target not in hot_targets:
            cold_by_target.setdefault(candidate.route_target, candidate)
    if local_candidate is not None and local_candidate.route_target not in hot_targets:
        cold_by_target.setdefault(local_candidate.route_target, local_candidate)

    cold = sorted(
        cold_by_target.values(),
        key=lambda candidate: _cold_rank(request_id, config.instance_id, candidate),
    )
    cold_budget = max(0, config.candidate_pool_size - len(hot))
    return CandidatePlan(tuple(hot), tuple(cold[:cold_budget]))


def build_candidate_pool(
    candidates: Sequence[KvcmCacheCandidate],
    local_candidate: Optional[KvcmCacheCandidate],
    config: KvcmFallbackConfig,
) -> list[KvcmCacheCandidate]:
    """Compatibility wrapper around :func:`build_candidate_plan`."""

    return list(
        build_candidate_plan(
            candidates,
            (),
            config,
            request_id="",
            local_candidate=local_candidate,
        ).candidates
    )


def _task_uncached_tokens(task: status_pb2.TaskInfoPB) -> int:
    input_tokens = max(0, int(task.input_length))
    hit_tokens = max(0, min(input_tokens, int(task.prefix_length)))
    if not task.is_waiting and task.HasField("remaining_prefill_tokens"):
        return max(0, int(task.remaining_prefill_tokens))
    return max(0, input_tokens - hit_tokens)


def worker_load_snapshot(
    candidate: KvcmCacheCandidate,
    status: status_pb2.WorkerStatusPB,
) -> WorkerLoadSnapshot:
    tasks = list(status.running_task_info)
    return WorkerLoadSnapshot(
        candidate=candidate,
        role=str(status.role),
        alive=bool(status.alive),
        waiting_task_count=sum(1 for task in tasks if task.is_waiting),
        outstanding_uncached_tokens=sum(_task_uncached_tokens(task) for task in tasks),
        status_version=int(status.status_version),
    )


def _matched_tokens(
    candidate: KvcmCacheCandidate,
    seq_len: int,
    block_size: int,
    p2p_hit_discount: float,
) -> int:
    # Java Math.round for a non-negative value is floor(value + 0.5).
    tokens = int(effective_cache_blocks(candidate, p2p_hit_discount) * block_size + 0.5)
    return min(max(0, seq_len), max(0, tokens))


def select_cache_affinity_first(
    snapshots: Sequence[WorkerLoadSnapshot],
    *,
    seq_len: int,
    config: KvcmFallbackConfig,
    last_selected_ns: Optional[Mapping[str, int]] = None,
) -> CacheAffinityDecision:
    """Run FlexLB CacheAffinityFirst semantics over the supplied subset."""

    recent = last_selected_ns or {}
    available: list[ScoredCandidate] = []
    for snapshot in snapshots:
        role = snapshot.role.upper()
        if (
            not snapshot.alive
            or ("PREFILL" not in role and "PDFUSION" not in role)
            or snapshot.waiting_task_count >= config.prefill_queue_size_threshold
        ):
            continue
        hit_tokens = _matched_tokens(
            snapshot.candidate,
            seq_len,
            config.block_size,
            config.p2p_hit_discount,
        )
        request_uncached = max(0, seq_len - hit_tokens)
        available.append(
            ScoredCandidate(
                snapshot=snapshot,
                hit_cache_tokens=hit_tokens,
                request_uncached_tokens=request_uncached,
                estimated_ttft_work=(
                    snapshot.outstanding_uncached_tokens + request_uncached
                ),
            )
        )

    if not available:
        return CacheAffinityDecision(None, "NO_AVAILABLE_WORKER", 0, 0)

    def ttft_key(worker: ScoredCandidate) -> tuple[int, int, str]:
        return (
            worker.estimated_ttft_work,
            recent.get(worker.candidate.route_target, 0),
            worker.candidate.route_target,
        )

    workers_by_ttft = sorted(available, key=ttft_key)
    threshold = config.outstanding_uncached_tokens_threshold
    eligible = (
        [
            worker
            for worker in workers_by_ttft
            if worker.snapshot.outstanding_uncached_tokens
            + worker.request_uncached_tokens
            <= threshold
        ]
        if threshold > 0
        else list(workers_by_ttft)
    )
    if not eligible:
        return CacheAffinityDecision(
            workers_by_ttft[0],
            "SHORTEST_TTFT_OUTSTANDING_GUARD_FALLBACK",
            len(available),
            0,
        )

    shortest = eligible[0]
    cache_leader = min(
        workers_by_ttft,
        key=lambda worker: (
            -worker.hit_cache_tokens,
            worker.estimated_ttft_work,
            recent.get(worker.candidate.route_target, 0),
            worker.candidate.route_target,
        ),
    )
    hit_rate_pct = (
        cache_leader.hit_cache_tokens * 100.0 / seq_len if seq_len > 0 else 0.0
    )
    if hit_rate_pct < config.cache_affinity_first_min_hit_rate:
        selected, reason = shortest, "SHORTEST_TTFT_LOW_CACHE_HIT"
    elif cache_leader not in eligible:
        selected, reason = shortest, "SHORTEST_TTFT_OUTSTANDING_GUARD"
    elif (
        cache_leader.estimated_ttft_work - shortest.estimated_ttft_work
        <= config.cache_affinity_first_max_extra_work_tokens
    ):
        selected, reason = cache_leader, "CACHE_LEADER"
    else:
        selected, reason = shortest, "SHORTEST_TTFT"
    return CacheAffinityDecision(selected, reason, len(available), len(eligible))


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
    """KVCM and WorkerStatus client with reusable channels and fresh probes."""

    def __init__(
        self,
        config: KvcmFallbackConfig,
        bootstrap_resolver: BootstrapResolver,
        candidate_snapshot_resolver: Optional[CandidateSnapshotResolver] = None,
    ) -> None:
        self.config = config
        self._bootstrap_resolver = bootstrap_resolver
        self._candidate_snapshot_resolver = candidate_snapshot_resolver
        self._channels: dict[str, grpc.aio.Channel] = {}
        self._stubs: dict[str, kvcm_pb2_grpc.MetaServiceStub] = {}
        self._worker_status_channels: dict[str, grpc.aio.Channel] = {}
        self._worker_status_stubs: dict[str, status_pb2_grpc.RpcServiceStub] = {}
        self._leader: Optional[str] = None
        self._leader_refreshed_at = 0.0
        self._leader_lock = asyncio.Lock()
        self._worker_status_semaphore = asyncio.Semaphore(
            self.config.worker_status_concurrency
        )
        self._logged_worker_hash_contract_mismatches: set[tuple[str, int, int]] = set()
        self._last_selected_ns: dict[str, int] = {}
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

    def _worker_status_stub_for(
        self,
        target: str,
    ) -> status_pb2_grpc.RpcServiceStub:
        if self._closed:
            raise KvcmFallbackError("KVCM fallback client is closed")
        stub = self._worker_status_stubs.get(target)
        if stub is not None:
            return stub
        channel = grpc.aio.insecure_channel(
            target,
            options=(
                ("grpc.keepalive_time_ms", 30_000),
                ("grpc.keepalive_timeout_ms", 5_000),
            ),
        )
        stub = status_pb2_grpc.RpcServiceStub(channel)
        self._worker_status_channels[target] = channel
        self._worker_status_stubs[target] = stub
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
            try:
                # Bootstrap discovery can use a synchronous VIP adapter. Keep it
                # off the routing event loop even when a caller does not provide
                # the preferred in-memory background snapshot resolver.
                resolved = await asyncio.to_thread(self._bootstrap_resolver)
            except Exception as error:
                raise KvcmFallbackError("KVCM bootstrap discovery failed") from error
            targets = tuple(dict.fromkeys(resolved))
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
            if previous_leader:
                self._leader_refreshed_at = time.monotonic()
                return previous_leader
            raise KvcmFallbackError("KVCM leader is unavailable")

    def _invalidate_leader(self, target: str) -> None:
        if self._leader == target:
            self._leader = None
            self._leader_refreshed_at = 0.0

    def _parse_candidates(
        self,
        hosts: Sequence[kvcm_pb2.HostCacheMatch],
    ) -> list[KvcmCacheCandidate]:
        candidates: list[KvcmCacheCandidate] = []
        for host in hosts:
            try:
                host_ip, http_port = _split_ip_port(host.host_ip_port)
            except (TypeError, ValueError):
                continue
            grpc_port = self.config.worker_grpc_port_override or http_port + 1
            status_port = self.config.worker_status_port_override or grpc_port
            if grpc_port > 65_535 or status_port > 65_535:
                continue
            candidates.append(
                KvcmCacheCandidate(
                    host_ip=host_ip,
                    http_port=http_port,
                    grpc_port=grpc_port,
                    worker_status_port=status_port,
                    local_blocks=max(0, int(host.local)),
                    p2p_fetch_blocks=max(0, int(host.p2p_1_fetch)),
                    p2p_total_match_blocks=max(0, int(host.p2p_1_total_match)),
                )
            )
        return candidates

    async def _query_cache_candidates(
        self,
        request_id: str,
        keys: Sequence[int],
    ) -> list[KvcmCacheCandidate]:
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
        return self._parse_candidates(response.hosts)

    async def _probe_worker(
        self,
        candidate: KvcmCacheCandidate,
    ) -> Optional[WorkerLoadSnapshot]:
        async with self._worker_status_semaphore:
            try:
                response = await self._worker_status_stub_for(
                    candidate.worker_status_target
                ).GetWorkerStatus(
                    status_pb2.StatusVersionPB(
                        latest_cache_version=-1,
                        latest_finished_version=-1,
                    ),
                    timeout=self.config.worker_status_timeout_ms / 1_000.0,
                )
            except (grpc.RpcError, asyncio.TimeoutError):
                return None
        reported_block_size = int(response.block_size)
        reported_lookahead_tokens = int(response.block_hash_lookahead_tokens)
        if (
            reported_block_size != self.config.block_size
            or reported_lookahead_tokens != self.config.block_hash_lookahead_tokens
        ):
            mismatch = (
                candidate.worker_status_target,
                reported_block_size,
                reported_lookahead_tokens,
            )
            if mismatch not in self._logged_worker_hash_contract_mismatches:
                self._logged_worker_hash_contract_mismatches.add(mismatch)
                route_logger.warning(
                    "WorkerStatus block hash contract mismatch; excluding fallback "
                    "worker, worker=%s, expected_block_size=%s, "
                    "expected_lookahead_tokens=%s, reported_block_size=%s, "
                    "reported_lookahead_tokens=%s",
                    candidate.worker_status_target,
                    self.config.block_size,
                    self.config.block_hash_lookahead_tokens,
                    reported_block_size,
                    reported_lookahead_tokens,
                )
            return None
        return worker_load_snapshot(candidate, response)

    async def query_and_select(
        self,
        *,
        request_id: str,
        block_cache_keys: Sequence[int],
        input_ids: Optional[Sequence[int]] = None,
        local_candidate: Optional[KvcmCacheCandidate] = None,
    ) -> KvcmFallbackResult:
        """Query KVCM once, then probe hot+cold candidates in bounded rounds."""

        started_at = time.monotonic_ns()
        keys = list(block_cache_keys)
        if not keys and input_ids is not None:
            keys = calculate_vllm_block_cache_keys(
                input_ids,
                self.config.block_size,
                self.config.block_hash_lookahead_tokens,
            )
        if keys:
            try:
                raw_candidates = await self._query_cache_candidates(request_id, keys)
            except Exception as error:
                route_logger.warning(
                    "KVCM cache query failed; continuing with discovered workers, "
                    "request_id=%s, error=%s",
                    request_id,
                    error,
                )
                raw_candidates = []
                cache_query_outcome = "query_failed"
            else:
                cache_query_outcome = (
                    "cache_candidates"
                    if any(
                        candidate.local_blocks >= self.config.minimum_local_blocks
                        or candidate.p2p_total_match_blocks
                        >= self.config.minimum_local_blocks
                        for candidate in raw_candidates
                    )
                    else "no_positive_match"
                )
        else:
            raw_candidates = []
            cache_query_outcome = "no_complete_blocks"

        discovered_candidates: list[KvcmCacheCandidate] = []
        if self._candidate_snapshot_resolver is not None:
            try:
                discovered_candidates = list(self._candidate_snapshot_resolver())
            except Exception as error:
                route_logger.warning(
                    "Fallback candidate snapshot read failed, request_id=%s, error=%s",
                    request_id,
                    error,
                )
        plan = build_candidate_plan(
            raw_candidates,
            discovered_candidates,
            self.config,
            request_id=request_id,
            local_candidate=local_candidate,
        )
        batches = plan.batches(self.config.cold_candidate_batch_size)
        if not batches:
            return KvcmFallbackResult(
                outcome="no_candidates",
                selected=None,
                candidate_count=len(raw_candidates),
                block_count=len(keys),
                latency_us=max(0, (time.monotonic_ns() - started_at) // 1_000),
                cache_query_outcome=cache_query_outcome,
                discovered_candidate_count=len(discovered_candidates),
            )

        status_started_at = time.monotonic_ns()
        snapshots_by_target: dict[str, WorkerLoadSnapshot] = {}
        decision = CacheAffinityDecision(None, "NO_AVAILABLE_WORKER", 0, 0)
        probe_round_count = 0
        seq_len = (
            len(input_ids)
            if input_ids is not None
            else len(keys) * self.config.block_size
        )
        for batch in batches:
            probe_round_count += 1
            probed = await asyncio.gather(
                *(self._probe_worker(candidate) for candidate in batch)
            )
            for snapshot in probed:
                if snapshot is not None:
                    snapshots_by_target[snapshot.candidate.route_target] = snapshot
            decision = select_cache_affinity_first(
                list(snapshots_by_target.values()),
                seq_len=seq_len,
                config=self.config,
                last_selected_ns=self._last_selected_ns,
            )
            if (
                decision.selected is not None
                and decision.reason != "SHORTEST_TTFT_OUTSTANDING_GUARD_FALLBACK"
            ):
                break

        status_latency_us = max(
            0,
            (time.monotonic_ns() - status_started_at) // 1_000,
        )
        scored = decision.selected
        if scored is not None:
            self._last_selected_ns[scored.candidate.route_target] = time.monotonic_ns()

        return KvcmFallbackResult(
            outcome="selected" if scored is not None else "no_available_worker",
            selected=None if scored is None else scored.candidate,
            candidate_count=len(raw_candidates),
            block_count=len(keys),
            latency_us=max(0, (time.monotonic_ns() - started_at) // 1_000),
            cache_query_outcome=cache_query_outcome,
            pool_candidate_count=len(plan.candidates),
            status_success_count=len(snapshots_by_target),
            status_latency_us=status_latency_us,
            selection_reason=decision.reason,
            selected_hit_cache_tokens=0 if scored is None else scored.hit_cache_tokens,
            selected_outstanding_uncached_tokens=(
                0 if scored is None else scored.snapshot.outstanding_uncached_tokens
            ),
            selected_request_uncached_tokens=(
                0 if scored is None else scored.request_uncached_tokens
            ),
            selected_estimated_ttft_work=(
                0 if scored is None else scored.estimated_ttft_work
            ),
            discovered_candidate_count=len(discovered_candidates),
            probe_round_count=probe_round_count,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        channels = [
            *self._channels.values(),
            *self._worker_status_channels.values(),
        ]
        self._channels.clear()
        self._stubs.clear()
        self._worker_status_channels.clear()
        self._worker_status_stubs.clear()
        self._logged_worker_hash_contract_mismatches.clear()
        for channel in channels:
            await channel.close()


__all__ = [
    "CandidatePlan",
    "CacheAffinityDecision",
    "KvcmCacheCandidate",
    "KvcmFallbackClient",
    "KvcmFallbackConfig",
    "KvcmFallbackError",
    "KvcmFallbackResult",
    "ScoredCandidate",
    "WorkerLoadSnapshot",
    "build_candidate_pool",
    "build_candidate_plan",
    "calculate_vllm_block_cache_keys",
    "effective_cache_blocks",
    "select_cache_affinity_first",
    "select_max_local_affinity",
    "worker_load_snapshot",
]
