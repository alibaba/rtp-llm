"""FlexLB schedule client: request role addrs from master/slave and parse response."""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp
from aiohttp import ClientTimeout

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import RoleAddr, RoleType
from rtp_llm.server.host_service import HostService
from rtp_llm.server.worker_status import ScheduleMeta
from rtp_llm.utils.base_model_datatypes import GenerateInput

route_logger = logging.getLogger("route_logger")

SCHEDULE_PATH = "/rtp_llm/schedule"
DEFAULT_REQUEST_TIMEOUT_SEC = 0.5
SUCCESS_CODE = 200
DEFAULT_REQUEST_PRIORITY = 100
CONNECTOR_LIMIT_PER_HOST = 30
CONNECTOR_KEEPALIVE_TIMEOUT_SEC = 30


@dataclass
class FlexlbResponse:
    """
    Result of a FlexLB schedule request: success or failure state.

    Success: role_addrs is set. Failure: connection_failed and/or
    error_code/error_message from scheduler. request_id is always from frontend;
    only connection_failed triggers slave retry and domain fallback.
    """

    role_addrs: Optional[List[RoleAddr]] = None
    connection_failed: bool = False
    error_code: Optional[int] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None  # internal: raw JSON from scheduler

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
    def ok(cls, role_addrs: List[RoleAddr]) -> "FlexlbResponse":
        """Business success: parsed role addrs."""
        return cls(
            role_addrs=role_addrs,
            connection_failed=False,
            error_code=None,
            error_message=None,
            result=None,
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

    def __init__(self, host_service=None, server_config=None, master_config=None):
        self.master_config = master_config
        self.host_service: Optional[HostService] = host_service
        self.max_connect_pool_size = (
            master_config.master_max_connect_pool_size if master_config else 1000
        )
        self._session: Optional[aiohttp.ClientSession] = None
        self.latest_queue_length: int = 0
        self.session_timeout_s = self._get_session_timeout_s()

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
    ) -> FlexlbResponse:
        """
        Resolve backend role addrs from FlexLB scheduler (master, then slave on connection failure).

        request_id is frontend-generated and only used for logging.
        Only connection_failed triggers slave retry and domain fallback.
        """
        master_addr = self.host_service.get_master_addr() if self.host_service else None
        if not master_addr:
            return FlexlbResponse.connection_failed_response()

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

        resp = await self._send_schedule_request(
            master_addr, payload, ttft_timeout_ms, request_id
        )

        if resp.connection_failed and slave_addr:
            route_logger.info(
                "Master connection failed, retrying slave, slave=%s, request_id=%s",
                slave_addr,
                request_id,
            )
            resp = await self._send_schedule_request(
                slave_addr, payload, ttft_timeout_ms, request_id
            )

        if resp.result is None:
            return FlexlbResponse(
                role_addrs=None,
                connection_failed=resp.connection_failed,
                error_code=resp.error_code,
                error_message=resp.error_message,
                result=None,
            )

        if resp.result.get("code", SUCCESS_CODE) != SUCCESS_CODE:
            raw_code = resp.result.get("code", SUCCESS_CODE)
            try:
                code = int(raw_code)
            except (TypeError, ValueError):
                code = int(ExceptionType.MASTER_NO_AVAILABLE_WORKER)
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
