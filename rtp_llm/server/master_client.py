"""FlexLB schedule client: request role addrs from master/slave and parse response."""

import inspect
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp
from aiohttp import ClientTimeout

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import RoleAddr, RoleType
from rtp_llm.server.host_service import HostService, VipServerWrapper
from rtp_llm.server.worker_status import ScheduleMeta
from rtp_llm.utils.base_model_datatypes import GenerateInput

route_logger = logging.getLogger("route_logger")

SCHEDULE_PATH = "/rtp_llm/schedule"
DEFAULT_REQUEST_TIMEOUT_SEC = 0.5
SUCCESS_CODE = 200
FALLBACK_ERROR_CODE = 8600
DEFAULT_REQUEST_PRIORITY = 100
CONNECTOR_LIMIT_PER_HOST = 30
CONNECTOR_KEEPALIVE_TIMEOUT_SEC = 30


@dataclass
class FlexlbResponse:
    """
    Result of a FlexLB schedule request: success or failure state.

    Success: role_addrs is set. Failure: connection_failed, fallback, and/or
    error_code/error_message from scheduler. request_id is always from frontend.
    Only connection_failed triggers slave retry; connection_failed and fallback
    both permit domain fallback.
    """

    role_addrs: Optional[List[RoleAddr]] = None
    connection_failed: bool = False
    fallback: bool = False
    error_code: Optional[int] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None  # internal: raw JSON from scheduler
    route_source: str = "NONE"
    cache_match: Optional[Dict[str, Any]] = None
    kvcm_outcome: Optional[str] = None

    @property
    def is_ok(self) -> bool:
        return self.role_addrs is not None

    @classmethod
    def ok_with_result(cls, result: Dict[str, Any]) -> "FlexlbResponse":
        """HTTP success: raw JSON body (parsed later into role_addrs)."""
        return cls(
            role_addrs=None,
            connection_failed=False,
            error_code=None,
            error_message=None,
            result=result,
        )

    @classmethod
    def ok(
        cls,
        role_addrs: List[RoleAddr],
        *,
        route_source: str = "FLEXLB",
        cache_match: Optional[Dict[str, Any]] = None,
        kvcm_outcome: Optional[str] = None,
    ) -> "FlexlbResponse":
        """Business success: parsed role addrs."""
        return cls(
            role_addrs=role_addrs,
            connection_failed=False,
            error_code=None,
            error_message=None,
            result=None,
            route_source=route_source,
            cache_match=cache_match,
            kvcm_outcome=kvcm_outcome,
        )

    @classmethod
    def error_response(
        cls,
        error_code: int,
        error_message: Optional[str] = None,
    ) -> "FlexlbResponse":
        """Scheduler returned error (e.g. non-200 body). No slave retry / no domain fallback."""
        return cls(
            role_addrs=None,
            connection_failed=False,
            error_code=error_code,
            error_message=error_message,
            result=None,
        )

    @classmethod
    def fallback_response(cls) -> "FlexlbResponse":
        return cls(
            role_addrs=None,
            connection_failed=False,
            fallback=True,
            error_code=FALLBACK_ERROR_CODE,
            error_message="FALLBACK",
            result=None,
        )

    @classmethod
    def connection_failed_response(cls) -> "FlexlbResponse":
        """No HTTP response (connection/timeout). Triggers slave retry and domain fallback."""
        return cls(
            role_addrs=None,
            connection_failed=True,
            error_code=None,
            error_message=None,
            result=None,
        )


class MasterClient:
    """Client for FlexLB schedule API (master and optional slave)."""

    def __init__(
        self,
        host_service=None,
        server_config=None,
        master_config=None,
        kvcm_fallback_client=None,
    ):
        self.master_config = master_config
        self.host_service: Optional[HostService] = host_service
        self.max_connect_pool_size = (
            master_config.master_max_connect_pool_size if master_config else 1000
        )
        self._session: Optional[aiohttp.ClientSession] = None
        self.latest_queue_length: int = 0
        self.session_timeout_s = self._get_session_timeout_s()
        self.client_fallback_enabled = bool(
            getattr(master_config, "master_client_fallback", False)
        )
        self._kvcm_vip = None
        self._kvcm_fallback_client = kvcm_fallback_client
        if self.client_fallback_enabled and self._kvcm_fallback_client is None:
            self._kvcm_fallback_client = self._create_kvcm_fallback_client()

    def _create_kvcm_fallback_client(self):
        """Create KVCM lazily only when the compatibility switch is enabled."""

        from rtp_llm.server.kvcm_fallback import KvcmFallbackClient, KvcmFallbackConfig

        service_id = str(
            getattr(self.master_config, "master_kvcm_service_id", "")
        ).strip()
        instance_id = str(
            getattr(self.master_config, "master_kvcm_instance_id", "")
        ).strip()
        bootstrap_port = int(
            getattr(self.master_config, "master_kvcm_bootstrap_port", 6381)
        )
        block_size = int(getattr(self.master_config, "master_kvcm_block_size", 0))
        request_timeout_ms = int(
            getattr(self.master_config, "master_kvcm_request_timeout_ms", 100)
        )
        grpc_port_override = int(
            getattr(
                self.master_config,
                "master_client_fallback_worker_grpc_port_override",
                0,
            )
        )
        worker_status_port = int(
            getattr(
                self.master_config,
                "master_client_fallback_worker_status_port",
                0,
            )
        )
        if not service_id:
            raise ValueError(
                "master_kvcm_service_id is required when KVCM fallback is enabled"
            )
        if not instance_id:
            raise ValueError(
                "master_kvcm_instance_id is required when KVCM fallback is enabled"
            )
        if not 1 <= bootstrap_port <= 65_535:
            raise ValueError("master_kvcm_bootstrap_port must be a valid port")

        self._kvcm_vip = VipServerWrapper(
            service_id,
            bool(getattr(self.master_config, "master_kvcm_use_local", False)),
        )

        def resolve_bootstrap_targets() -> List[str]:
            targets: List[str] = []
            for host in self._kvcm_vip.get_hosts(refresh=True):
                ip = str(host.ip)
                if ":" in ip and not ip.startswith("["):
                    targets.append(f"[{ip}]:{bootstrap_port}")
                else:
                    targets.append(f"{ip}:{bootstrap_port}")
            return targets

        return KvcmFallbackClient(
            KvcmFallbackConfig(
                instance_id=instance_id,
                block_size=block_size,
                request_timeout_ms=request_timeout_ms,
                worker_grpc_port_override=(grpc_port_override or None),
                worker_status_port_override=(worker_status_port or None),
                candidate_pool_size=int(
                    getattr(
                        self.master_config,
                        "master_client_fallback_candidate_pool_size",
                        3,
                    )
                ),
                hot_candidate_pool_size=int(
                    getattr(
                        self.master_config,
                        "master_kvcm_hot_candidate_pool_size",
                        2,
                    )
                ),
                worker_status_concurrency=int(
                    getattr(
                        self.master_config,
                        "master_client_fallback_worker_status_concurrency",
                        3,
                    )
                ),
                worker_status_timeout_ms=int(
                    getattr(
                        self.master_config,
                        "master_client_fallback_worker_status_timeout_ms",
                        200,
                    )
                ),
                prefill_queue_size_threshold=int(
                    getattr(
                        self.master_config,
                        "master_client_fallback_prefill_queue_size_threshold",
                        1024,
                    )
                ),
                p2p_hit_discount=float(
                    getattr(
                        self.master_config,
                        "master_client_fallback_p2p_hit_discount",
                        0.2,
                    )
                ),
                cache_affinity_first_max_extra_work_tokens=int(
                    getattr(
                        self.master_config,
                        "master_client_fallback_cache_affinity_first_max_extra_work_tokens",
                        0,
                    )
                ),
                outstanding_uncached_tokens_threshold=int(
                    getattr(
                        self.master_config,
                        "master_client_fallback_outstanding_uncached_tokens_threshold",
                        0,
                    )
                ),
                cache_affinity_first_min_hit_rate=float(
                    getattr(
                        self.master_config,
                        "master_client_fallback_cache_affinity_first_min_hit_rate",
                        5.0,
                    )
                ),
            ),
            resolve_bootstrap_targets,
        )

    def _get_session_timeout_s(self) -> float:
        # Session-level timeout is a safety net for the connection pool lifetime,
        # not for individual requests. Per-request timeout in _send_schedule_request
        # always takes precedence (aiohttp per-request timeout overrides session timeout).
        if self.master_config and self.master_config.master_session_timeout_s >= 0:
            return float(self.master_config.master_session_timeout_s)
        if self.host_service and self.host_service.master_vip.domain:
            return 3600.0
        return DEFAULT_REQUEST_TIMEOUT_SEC

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = ClientTimeout(total=self.session_timeout_s)
            connector = aiohttp.TCPConnector(
                limit=self.max_connect_pool_size,
                limit_per_host=CONNECTOR_LIMIT_PER_HOST,
                keepalive_timeout=CONNECTOR_KEEPALIVE_TIMEOUT_SEC,
                enable_cleanup_closed=True,
            )
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        if self._kvcm_fallback_client is not None:
            close_result = self._kvcm_fallback_client.close()
            if inspect.isawaitable(close_result):
                await close_result

    @staticmethod
    def _input_ids_for_kvcm(input: GenerateInput) -> Optional[List[int]]:
        raw_input_ids = getattr(input, "input_ids", None)
        if raw_input_ids is None:
            raw_input_ids = getattr(input, "token_ids", None)
        if raw_input_ids is None:
            return None
        if hasattr(raw_input_ids, "tolist"):
            raw_input_ids = raw_input_ids.tolist()
        if (
            isinstance(raw_input_ids, list)
            and len(raw_input_ids) == 1
            and isinstance(raw_input_ids[0], list)
        ):
            raw_input_ids = raw_input_ids[0]
        if not isinstance(raw_input_ids, list):
            return None
        try:
            return [int(token_id) for token_id in raw_input_ids]
        except (TypeError, ValueError):
            return None

    async def _try_kvcm_fallback(
        self,
        block_cache_keys: list[int],
        input: GenerateInput,
        request_id: int,
        local_fallback_addr: Optional[RoleAddr] = None,
    ) -> Optional[FlexlbResponse]:
        if not self.client_fallback_enabled or self._kvcm_fallback_client is None:
            return None

        local_candidate = None
        if local_fallback_addr is not None:
            from rtp_llm.server.kvcm_fallback import KvcmCacheCandidate

            status_port = int(
                getattr(
                    self.master_config,
                    "master_client_fallback_worker_status_port",
                    0,
                )
            ) or int(local_fallback_addr.grpc_port)
            local_candidate = KvcmCacheCandidate(
                host_ip=str(local_fallback_addr.ip),
                http_port=int(local_fallback_addr.http_port),
                grpc_port=int(local_fallback_addr.grpc_port),
                worker_status_port=status_port,
                local_blocks=0,
                p2p_fetch_blocks=0,
                p2p_total_match_blocks=0,
            )

        try:
            result = await self._kvcm_fallback_client.query_and_select(
                request_id=str(request_id),
                block_cache_keys=block_cache_keys,
                input_ids=self._input_ids_for_kvcm(input),
                local_candidate=local_candidate,
            )
        except Exception as error:
            route_logger.warning(
                "KVCM fallback failed, request_id=%s, error=%s",
                request_id,
                error,
            )
            return None

        selected = result.selected
        if selected is None:
            route_logger.info(
                "KVCM fallback returned no route, request_id=%s, outcome=%s, "
                "candidate_count=%s, latency_us=%s",
                request_id,
                result.outcome,
                result.candidate_count,
                result.latency_us,
            )
            return None

        cache_match = {
            "host": selected.host_ip_port,
            "local_blocks": selected.local_blocks,
            "p2p_fetch_blocks": selected.p2p_fetch_blocks,
            "p2p_total_match_blocks": selected.p2p_total_match_blocks,
            "block_count": result.block_count,
            "candidate_count": result.candidate_count,
            "pool_candidate_count": getattr(result, "pool_candidate_count", 0),
            "status_success_count": getattr(result, "status_success_count", 0),
            "status_latency_us": getattr(result, "status_latency_us", 0),
            "selection_reason": getattr(result, "selection_reason", None),
            "hit_cache_tokens": getattr(result, "selected_hit_cache_tokens", 0),
            "outstanding_uncached_tokens": getattr(
                result, "selected_outstanding_uncached_tokens", 0
            ),
            "request_uncached_tokens": getattr(
                result, "selected_request_uncached_tokens", 0
            ),
            "estimated_ttft_work": getattr(result, "selected_estimated_ttft_work", 0),
            "latency_us": result.latency_us,
        }
        route_logger.info(
            "KVCM fallback selected worker, request_id=%s, worker=%s, "
            "local_blocks=%s, candidate_count=%s, pool_candidate_count=%s, "
            "status_success_count=%s, selection_reason=%s, latency_us=%s",
            request_id,
            selected.host_ip_port,
            selected.local_blocks,
            result.candidate_count,
            getattr(result, "pool_candidate_count", 0),
            getattr(result, "status_success_count", 0),
            getattr(result, "selection_reason", None),
            result.latency_us,
        )
        return FlexlbResponse.ok(
            [
                RoleAddr(
                    role=RoleType.PREFILL,
                    ip=selected.host_ip,
                    http_port=selected.http_port,
                    grpc_port=selected.grpc_port,
                )
            ],
            route_source="KVCM",
            cache_match=cache_match,
            kvcm_outcome=result.outcome,
        )

    def get_latest_queue_length(self) -> int:
        return self.latest_queue_length

    async def _send_schedule_request(
        self,
        addr: str,
        payload: Dict[str, Any],
        generate_timeout_ms: int,
        request_id: int,
    ) -> FlexlbResponse:
        """
        Send one schedule request to the given host (master or slave).
        Returns FlexlbResponse: ok_with_result on HTTP success, error_response on
        non-200 body, connection_failed_response when no response received.
        """
        url = f"http://{addr}{SCHEDULE_PATH}"
        headers = {"Content-Type": "application/json"}
        timeout_sec = (
            (generate_timeout_ms / 1000.0)
            if generate_timeout_ms > 0
            else DEFAULT_REQUEST_TIMEOUT_SEC
        )
        start = time.time()

        try:
            session = await self._get_session()
            request_timeout = ClientTimeout(total=timeout_sec)
            async with session.post(
                url,
                data=json.dumps(payload),
                headers=headers,
                timeout=request_timeout,
            ) as response:
                if response.status != SUCCESS_CODE:
                    error_code = int(ExceptionType.MASTER_NO_AVAILABLE_WORKER)
                    error_message = None
                    try:
                        raw = await response.json()
                        if isinstance(raw, dict):
                            raw_code = raw.get("code")
                            if raw_code is not None:
                                try:
                                    error_code = int(raw_code)
                                except (TypeError, ValueError):
                                    pass
                            error_message = raw.get("error_message")
                    except (json.JSONDecodeError, aiohttp.ClientError):
                        pass
                    route_logger.error(
                        "FlexLB schedule failed, request_id=%s, error_code=%s, error_message=%s",
                        request_id,
                        error_code,
                        error_message or "",
                    )
                    if error_code == FALLBACK_ERROR_CODE:
                        return FlexlbResponse.fallback_response()
                    return FlexlbResponse.error_response(error_code, error_message)

                result = await response.json()
                return FlexlbResponse.ok_with_result(result)

        except (aiohttp.ClientError, TimeoutError, ConnectionError, OSError) as e:
            elapsed = time.time() - start
            route_logger.error(
                "Schedule request failed, addr=%s, request_id=%s, error=%s, elapsed=%.3fs",
                addr,
                request_id,
                e,
                elapsed,
            )
            return FlexlbResponse.connection_failed_response()
        except Exception as e:
            elapsed = time.time() - start
            route_logger.exception(
                "Unexpected error in schedule request, addr=%s, request_id=%s, elapsed=%.3fs",
                addr,
                request_id,
                elapsed,
            )
            return FlexlbResponse.connection_failed_response()

    async def get_backend_role_addrs(
        self,
        block_cache_keys: list[int],
        input: GenerateInput,
        request_id: int,
        local_fallback_addr: Optional[RoleAddr] = None,
    ) -> FlexlbResponse:
        """
        Resolve backend role addrs from FlexLB scheduler (master, then slave on connection failure).

        request_id is frontend-generated and only used for logging.
        Only connection_failed triggers slave retry. A fallback response is returned
        directly so the caller can perform domain fallback.
        """
        master_addr = self.host_service.get_master_addr() if self.host_service else None
        slave_addr = None
        if self.host_service:
            slave_addr = getattr(self.host_service, "get_slave_addr", lambda: None)()

        ttft_timeout_ms = getattr(
            input.generate_config, "ttft_timeout_ms", None
        ) or getattr(input.generate_config, "timeout_ms", None)
        if not ttft_timeout_ms or ttft_timeout_ms <= 0:
            ttft_timeout_ms = (
                self.master_config.master_default_timeout_ms
                if self.master_config
                else 3600000
            )
        request_priority = getattr(
            input.generate_config,
            "traffic_reject_priority",
            DEFAULT_REQUEST_PRIORITY,
        )
        start = time.time()

        payload: Dict[str, Any] = {
            "model": "engine_service",
            "block_cache_keys": block_cache_keys,
            "seq_len": input.prompt_length,
            "debug": False,
            "request_priority": request_priority,
            "generate_timeout": ttft_timeout_ms,
            "request_id": request_id,
            "request_time_ms": int(start * 1000),
        }

        flexlb_timeout_ms = ttft_timeout_ms
        configured_transport_timeout_ms = int(
            getattr(
                self.master_config,
                "master_client_fallback_flexlb_transport_timeout_ms",
                0,
            )
            if self.master_config
            else 0
        )
        if configured_transport_timeout_ms > 0:
            flexlb_timeout_ms = min(
                flexlb_timeout_ms,
                configured_transport_timeout_ms,
            )

        resp = FlexlbResponse.connection_failed_response()
        if master_addr:
            resp = await self._send_schedule_request(
                master_addr, payload, flexlb_timeout_ms, request_id
            )

        if resp.connection_failed and slave_addr:
            route_logger.info(
                "Master connection failed, retrying slave, slave=%s, request_id=%s",
                slave_addr,
                request_id,
            )
            resp = await self._send_schedule_request(
                slave_addr, payload, flexlb_timeout_ms, request_id
            )

        # KVCM is an availability fallback only.  Explicit FlexLB fallback
        # (8600) and all business/admission errors keep their original meaning.
        if resp.connection_failed:
            kvcm_response = await self._try_kvcm_fallback(
                block_cache_keys,
                input,
                request_id,
                local_fallback_addr,
            )
            if kvcm_response is not None:
                return kvcm_response

        if resp.result is None:
            return FlexlbResponse(
                role_addrs=None,
                connection_failed=resp.connection_failed,
                fallback=resp.fallback,
                error_code=resp.error_code,
                error_message=resp.error_message,
                result=None,
                route_source=resp.route_source,
                cache_match=resp.cache_match,
                kvcm_outcome=resp.kvcm_outcome,
            )

        if resp.result.get("code", SUCCESS_CODE) != SUCCESS_CODE:
            raw_code = resp.result.get("code", SUCCESS_CODE)
            try:
                code = int(raw_code)
            except (TypeError, ValueError):
                code = int(ExceptionType.MASTER_NO_AVAILABLE_WORKER)
            if code == FALLBACK_ERROR_CODE:
                return FlexlbResponse.fallback_response()
            try:
                exception_type = ExceptionType(code)
            except ValueError:
                exception_type = ExceptionType.MASTER_NO_AVAILABLE_WORKER
            message = resp.result.get("error_message") or "master schedule error"
            route_logger.error(
                "Master schedule error, request_id=%s, error_code=%s, error_message=%s",
                request_id,
                code,
                message,
            )
            raise FtRuntimeException(exception_type=exception_type, message=message)

        schedule_meta = ScheduleMeta.model_validate(resp.result)
        role_addrs = [
            RoleAddr(
                role=RoleType(s.role),  # type: ignore[arg-type]
                ip=s.server_ip,
                http_port=s.http_port,
                grpc_port=s.grpc_port,
            )
            for s in schedule_meta.server_status
        ]
        return FlexlbResponse.ok(role_addrs)
