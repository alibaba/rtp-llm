"""PP-only startup glue for the KV Collector; no cache polling RPCs."""

import functools
import json
import logging
import os
import re
import sys
import threading
from dataclasses import dataclass
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PPCollectorOptions:
    group: str
    port: int
    stages: int
    shard_id: int


def _integer(environ, name):
    value = environ.get(name, "")
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise ValueError(f"{name} must be an unsigned decimal integer")
    return int(value)


def pp_collector_options(environ, *, actual_pp_size=None):
    if actual_pp_size is not None and actual_pp_size <= 1:
        return None
    enabled = environ.get("NAVI_SCHED_PP_CACHE_COLLECTOR", "0")
    if enabled == "0":
        return None
    if enabled != "1":
        raise ValueError("NAVI_SCHED_PP_CACHE_COLLECTOR must be 0 or 1")
    stages = _integer(environ, "NAVI_SCHED_PP_SIZE")
    shard = _integer(environ, "NAVI_SCHED_PP_SHARD_ID")
    port = _integer(environ, "NAVI_SCHED_PP_CACHE_COLLECTOR_PORT")
    group = environ.get("NAVI_SCHED_PP_CACHE_COLLECTOR_GROUP", "")
    if stages <= 1 or (actual_pp_size is not None and stages != actual_pp_size):
        raise ValueError(
            "PP Collector stages must match pipeline parallelism greater than one"
        )
    if not 0 <= shard < stages or not 0 < port <= 65535:
        raise ValueError("invalid PP Collector shard or port")
    if not isinstance(group, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", group):
        raise ValueError("invalid PP Collector group")
    return PPCollectorOptions(group, port, stages, shard)


def make_pp_cache_agent_config(scheduler, vllm_config, core, *, environ=None):
    environ = os.environ if environ is None else environ
    if environ.get("NAVI_SCHED_PP_CACHE_COLLECTOR", "0") == "0":
        return None
    parallel = vllm_config.parallel_config
    stages = int(parallel.pipeline_parallel_size)
    options = pp_collector_options(environ, actual_pp_size=stages)
    if options is None:
        return None
    if options.shard_id != 0:
        raise ValueError("PP cache Agent must run on pipeline stage zero")
    if parallel.tensor_parallel_size != 1 or parallel.nnodes_within_dp != stages:
        raise ValueError(
            "PP V6D Collector requires one pipeline stage per node and TP=1"
        )
    module = sys.modules.get("kvs_connector.v6d_object_connector")
    connector_type = getattr(module, "V6dObjectConnector", None)
    connectors = [getattr(scheduler, "connector", None)]
    visited = set()
    source = None
    while connectors:
        connector = connectors.pop()
        if connector is None or id(connector) in visited:
            continue
        visited.add(id(connector))
        if connector_type is not None and isinstance(connector, connector_type):
            source = connector.scheduler
            break
        connectors.extend(getattr(connector, "_connectors", ()))
    if source is None or source._pp_size != stages:
        raise ValueError("PP Collector requires the native PP V6D scheduler")
    urls = tuple(source.v6d_urls)
    configured_hosts = None
    raw_hosts = environ.get("NAVI_SCHED_PP_CACHE_COLLECTOR_HOSTS")
    if raw_hosts is not None:
        configured_hosts = json.loads(raw_hosts)
        if (
            not isinstance(configured_hosts, list)
            or len(configured_hosts) != stages
            or any(
                not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", host)
                for host in configured_hosts
            )
        ):
            raise ValueError("PP Collector hosts must name every pipeline node")
    if urls:
        if len(urls) != stages:
            raise ValueError("PP V6D worker handshake has not supplied every node")
        hosts = []
        for url in urls:
            address = urlsplit(url)
            if (
                address.scheme not in ("http", "https")
                or not address.hostname
                or address.username is not None
                or address.password is not None
                or address.port == options.port
            ):
                raise ValueError(
                    "invalid PP V6D node address or colliding Collector port"
                )
            hosts.append(address.hostname)
        if configured_hosts is not None and hosts != configured_hosts:
            raise ValueError(
                "configured PP Collector hosts differ from the native handshake"
            )
    else:
        # Agent creation precedes the native worker handshake. Explicit rollout
        # addresses are verified again when EngineCore binds after that handshake.
        if configured_hosts is None:
            raise ValueError(
                "PP handshake is pending; explicit Collector hosts are required"
            )
        hosts = configured_hosts
        if options.port == getattr(source, "v6d_port", None):
            raise ValueError("PP Collector port collides with native V6D")
    config = core.PPCacheAgentConfig()
    config.pp_group = options.group
    config.pipeline_parallel_size = stages
    endpoints = []
    for shard, host in enumerate(hosts):
        endpoint = core.PPCacheShardEndpoint()
        endpoint.shard_id = shard
        endpoint.biz_name = "navi_sched_pp_cache_collector"
        endpoint.host = host
        endpoint.grpc_port = options.port
        endpoints.append(endpoint)
    config.shards = endpoints
    native = source.vllm_config
    cache = native.cache_config
    eagle = (
        native.speculative_config is not None and native.speculative_config.use_eagle()
    )
    hybrid = bool(source.mamba_group_ids)
    align = hybrid and cache.mamba_cache_mode == "align"
    if source.hash_block_size <= 0 or source.lcm_block_size % source.hash_block_size:
        raise ValueError("invalid native V6D hash block alignment")
    policy = config.policy
    policy.block_size = source.hash_block_size
    policy.linear_step = source.lcm_block_size // source.hash_block_size
    policy.hybrid_tail_required = hybrid
    policy.full_attention_query_drop_tokens = int(eagle)
    policy.mamba_replay_stop_shift = bool(align and eagle)
    policy.mamba_retention_interval = (
        -1
        if cache.prefix_cache_retention_interval is None
        else int(cache.prefix_cache_retention_interval)
    )
    policy.mamba_align = align
    if not config.valid():
        raise ValueError(
            "invalid PP Collector topology or independent V6D cache policy"
        )
    return config


def verify_pp_cache_handshake(config, scheduler):
    module = sys.modules.get("kvs_connector.v6d_object_connector")
    connector_type = getattr(module, "V6dObjectConnector", None)
    connectors = [getattr(scheduler, "connector", None)]
    visited = set()
    while connectors:
        connector = connectors.pop()
        if connector is None or id(connector) in visited:
            continue
        visited.add(id(connector))
        if connector_type is not None and isinstance(connector, connector_type):
            source = connector.scheduler
            urls = tuple(source.v6d_urls)
            hosts = [urlsplit(url).hostname for url in urls]
            expected = [
                node.host
                for node in sorted(config.shards, key=lambda node: node.shard_id)
            ]
            if source._pp_size != config.pipeline_parallel_size or hosts != expected:
                raise ValueError(
                    "PP Collector topology does not match the completed native handshake"
                )
            return
        connectors.extend(getattr(connector, "_connectors", ()))
    raise ValueError("native PP V6D connector is absent after worker handshake")


def start_pp_daemon_collector(daemon, options, core, *, invalid_object_id):
    from .V6DCollectorObserver import V6DCollectorObserver

    spec = daemon._kv_cache_spec
    if spec is None or spec.get("node_count", 1) != options.stages:
        raise ValueError("native V6D spec does not match the PP node count")
    groups = sorted(spec["groups"], key=lambda group: group["group_id"])
    if [group["group_id"] for group in groups] != list(
        range(len(groups))
    ) or not groups:
        raise ValueError("native V6D group IDs must be complete and ordered")
    roles = []
    for group in groups:
        if group["spec_type"] == "FullAttentionSpec":
            roles.append(core.PPCacheGroupRole.FULL)
        elif group["spec_type"] == "MambaSpec":
            roles.append(core.PPCacheGroupRole.TAIL)
        else:
            raise ValueError("unsupported native PP V6D cache group type")
    config = core.NaviSchedCollectorConfig()
    config.pp_group = options.group
    config.shard_id = options.shard_id
    config.shard_count = options.stages
    config.group_roles = roles
    # Native V6D metadata already expands each object to 32-byte hash-block keys.
    config.group_block_scales = [1] * len(roles)
    config.service.name = f"pp_cache_{options.group}_{options.shard_id}"
    config.service.enable_kmon = False
    config.service.enable_snapshot = False
    config.service.wait_no_query = False
    config.service.gig.grpc_port = options.port
    config.service.gig.grpc_io_thread_num = 1
    config.runtime.biz_name = "navi_sched_pp_cache_collector"
    config.runtime.thread_num = 2
    collector = core.NaviSchedCollector(config)
    try:
        if not collector.init():
            raise RuntimeError("failed to initialize PP KV Collector")
        observer = V6DCollectorObserver(
            daemon.peer,
            collector,
            pp_shard_count=options.stages,
            invalid_object_id=invalid_object_id,
        ).install()
    except BaseException:
        collector.stop()
        raise
    daemon._navi_sched_pp_collector = observer
    return observer


def install_pp_daemon_hook(
    daemon_type, *, environ=None, core=None, invalid_object_id=None
):
    options = pp_collector_options(os.environ if environ is None else environ)
    if options is None:
        return False
    if getattr(daemon_type, "_navi_sched_pp_hook", None) is not None:
        raise RuntimeError("PP V6D daemon hook is already installed")
    original = daemon_type.push_kv_cache_spec

    @functools.wraps(original)
    async def observed(daemon, *args, **kwargs):
        result = await original(daemon, *args, **kwargs)
        if getattr(daemon, "_navi_sched_pp_collector", None) is not None:
            return result
        try:
            native_core = core
            if native_core is None:
                from navi_sched.util import naviSchedModule

                native_core = naviSchedModule()
            invalid = invalid_object_id
            if invalid is None:
                from v6d.lite.common.type import ObjectID

                invalid = ObjectID.invalid_object_id().id
            start_pp_daemon_collector(
                daemon, options, native_core, invalid_object_id=invalid
            )
            daemon._navi_sched_pp_collector_error = False
        except Exception:
            # Native spec handling remains authoritative. Rollout readiness must
            # reject a missing Collector; observation never replaces its result.
            daemon._navi_sched_pp_collector_error = True
            logger.exception("failed to start PP KV Collector after native V6D spec")
        return result

    daemon_type.push_kv_cache_spec = observed
    daemon_type._navi_sched_pp_hook = options
    return True


class PPRequestCompletionObserver:
    """Forward native request outcomes; asynchronous KV seals remain independent."""

    def __init__(self, agent, core, finish_reasons):
        self._agent = agent
        self._status = core.PPCacheRequestStatus
        self._finish_reasons = finish_reasons
        self._requests = {}
        self._closed = False
        self._lock = threading.Lock()

    def begin(self, external_id, request_id):
        external_id = str(external_id)
        try:
            with self._lock:
                if self._closed or external_id in self._requests:
                    return None
                attempt = self._agent.begin_pp_cache_request(request_id)
                if not attempt:
                    return None
                ticket = (request_id, attempt)
                self._requests[external_id] = ticket
                return ticket
        except Exception:
            logger.exception("failed to register PP request completion")
            return None

    def _finish(self, external_id, status, expected=None):
        with self._lock:
            ticket = self._requests.get(str(external_id))
            if ticket is None or (expected is not None and ticket != expected):
                return False
            del self._requests[str(external_id)]
        try:
            return self._agent.finish_pp_cache_request(*ticket, status)
        except Exception:
            logger.exception("failed to publish PP request completion")
            return False

    def capture_requests(self, request_ids):
        captured = []
        with self._lock:
            for request_id in request_ids:
                external_id = str(request_id)
                ticket = self._requests.get(external_id)
                if ticket is not None:
                    captured.append((external_id, ticket))
        return tuple(captured)

    def finish_execution_error(self, requests, *, cancelled):
        status = self._status.CANCELLED if cancelled else self._status.FAILED
        for external_id, ticket in requests:
            self._finish(external_id, status, ticket)

    def fail_admission(self, external_id, ticket):
        if ticket is not None:
            self._finish(external_id, self._status.FAILED, ticket)

    def finish_requests(self, request_ids, finished_status):
        name = getattr(finished_status, "name", "")
        status = self._status.FAILED
        if name == "FINISHED_ABORTED":
            status = self._status.CANCELLED
        elif name in ("FINISHED_STOPPED", "FINISHED_LENGTH_CAPPED"):
            status = self._status.SUCCEEDED
        self._finish_many(request_ids, status)

    def cancel_requests(self, request_ids):
        self._finish_many(request_ids, self._status.CANCELLED)

    def fail_requests(self, request_ids):
        self._finish_many(request_ids, self._status.FAILED)

    def _finish_many(self, request_ids, status):
        if request_ids is None:
            with self._lock:
                requests = tuple(self._requests.items())
            for external_id, ticket in requests:
                self._finish(external_id, status, ticket)
        else:
            for external_id in request_ids:
                self._finish(external_id, status)

    def observe_outputs(self, outputs):
        results = {}
        for output in outputs.outputs or ():
            if not output.finished:
                continue
            reason = output.finish_reason
            status = self._status.FAILED
            if reason in (self._finish_reasons.STOP, self._finish_reasons.LENGTH):
                status = self._status.SUCCEEDED
            elif reason == self._finish_reasons.ABORT:
                status = self._status.CANCELLED
            results[str(output.request_id)] = status
        for external_id in outputs.finished_requests or ():
            # Native free notifications without a result cannot prove success.
            results.setdefault(str(external_id), self._status.FAILED)
        for external_id, status in results.items():
            self._finish(external_id, status)

    def close(self):
        with self._lock:
            self._closed = True
        self.fail_requests(None)
