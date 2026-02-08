import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from aiohttp import ClientTimeout

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import RoleAddr, RoleType
from rtp_llm.server.worker_status import ScheduleMeta
from rtp_llm.utils.base_model_datatypes import GenerateInput

route_logger = logging.getLogger("route_logger")


class MasterClient:
    def __init__(self, host_service=None, server_config=None, master_config=None):
        self.master_config = master_config
        self.max_connect_pool_size = (
            master_config.master_max_connect_pool_size if master_config else 100000
        )
        self._session = None
        self.latest_queue_length: int = 0
        self.host_service = host_service
        self.session_timeout_s = self._get_session_timeout_s()

    def _get_session_timeout_s(self) -> float:
        """Session total timeout in seconds. If master_session_timeout_s >= 0 use it; else 3600 when queue mode, 0.5 otherwise."""
        if self.master_config and self.master_config.master_session_timeout_s >= 0:
            return float(self.master_config.master_session_timeout_s)
        # Queue mode: master domain is set
        if self.host_service and self.host_service.master_vip.domain:
            return 3600.0
        return 0.5

    async def _get_session(self):
        """获取或创建HTTP session"""
        if self._session is None or self._session.closed:
            timeout = ClientTimeout(total=self.session_timeout_s)
            connector = aiohttp.TCPConnector(
                limit=self.max_connect_pool_size,
                keepalive_timeout=30,
                enable_cleanup_closed=True,
            )
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self._session

    async def close(self):
        """关闭HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()

    def get_latest_queue_length(self) -> int:
        return self.latest_queue_length

    async def _send_schedule_request(
        self,
        addr: str,
        payload: Dict[str, Any],
        generate_timeout: int,
        request_id: int,
        start: float,
    ) -> Optional[Dict[str, Any]]:
        """Send schedule request to given address, return result or None on failure."""
        url = f"http://{addr}/rtp_llm/schedule"
        headers = {"Content-Type": "application/json"}

        try:
            session = await self._get_session()
            request_timeout = ClientTimeout(total=generate_timeout / 1000.0)
            async with session.post(
                url, data=json.dumps(payload), headers=headers, timeout=request_timeout
            ) as response:
                if response.status != 200:
                    route_logger.error(
                        f"Failed to get response from {addr}, http status: {response.status}, request_id: {request_id}"
                    )
                    return None
                return await response.json()
        except Exception as e:
            rt = time.time() - start
            route_logger.error(
                f"Request to {addr} failed, request_id: {request_id}, error: {e}, rt={rt:.3f}s"
            )
            return None

    async def get_backend_role_addrs(
        self, block_cache_keys: list[int], input: GenerateInput, request_id: int
    ) -> Optional[List[RoleAddr]]:
        master_addr = self.host_service.get_master_addr()
        route_logger.debug(f"routing to master: {master_addr}")
        if not master_addr:
            return None

        seq_len = input.prompt_length
        ttft_timeout_ms = input.generate_config.ttft_timeout_ms
        timeout_ms = input.generate_config.timeout_ms
        start = time.time()
        generate_timeout = (
            ttft_timeout_ms
            if ttft_timeout_ms and ttft_timeout_ms > 0
            else (
                timeout_ms
                if timeout_ms and timeout_ms > 0
                else (
                    self.master_config.master_default_timeout_ms
                    if self.master_config
                    else 3600000
                )
            )
        )
        request_priority = input.generate_config.traffic_reject_priority

        payload = {
            "model": "engine_service",
            "block_cache_keys": block_cache_keys,
            "seq_len": seq_len,
            "debug": False,
            "request_priority": request_priority,
            "generate_timeout": generate_timeout,
            "request_id": request_id,
            "request_time_ms": int(start * 1000),
        }

        result = await self._send_schedule_request(
            master_addr, payload, generate_timeout, request_id, start
        )
        if result is None:
            slave_addr = self.host_service.get_slave_addr()
            if slave_addr:
                route_logger.info(
                    f"Retrying with slave node: {slave_addr}, request_id: {request_id}"
                )
                result = await self._send_schedule_request(
                    slave_addr, payload, generate_timeout, request_id, start
                )
            if result is None:
                return None

        # check response
        schedule_meta = ScheduleMeta.model_validate(result)
        if schedule_meta.code != 200:
            route_logger.error(
                f"Master schedule error, error code: {schedule_meta.code}"
            )
            raise FtRuntimeException(
                exception_type=ExceptionType(schedule_meta.code),
                message="master schedule error",
            )

        # parse role ips from schedule meta
        role_addrs: List[RoleAddr] = []
        for server_status in schedule_meta.server_status:
            role_addrs.append(
                RoleAddr(
                    role=RoleType(server_status.role),
                    ip=server_status.server_ip,
                    http_port=server_status.http_port,
                    grpc_port=server_status.grpc_port,
                )
            )

        return role_addrs
