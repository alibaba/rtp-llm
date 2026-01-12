import json
import logging
from typing import List, Optional, Tuple

import aiohttp
from aiohttp import ClientTimeout

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import RoleAddr, RoleType
from rtp_llm.server.worker_status import ScheduleMeta

route_logger = logging.getLogger("route_logger")


class MasterClient:
    def __init__(self, max_connect_pool_size=1000):
        self.max_connect_pool_size = max_connect_pool_size
        self._session = None

    async def _get_session(self):
        """获取或创建HTTP session"""
        if self._session is None or self._session.closed:
            timeout = ClientTimeout(total=0.5)
            connector = aiohttp.TCPConnector(
                limit=self.max_connect_pool_size,  # con pool size
                limit_per_host=30,  # limit
                keepalive_timeout=30,
                enable_cleanup_closed=True,
            )
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self._session

    async def close(self):
        """关闭HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_backend_role_addrs(
        self,
        master_addr: Optional[str],
        block_cache_keys: list[int],
        seq_len: int,
        debug: bool,
        generate_timeout: int,
        request_id: int,
        request_priority: int = 100,
    ) -> Tuple[Optional[List[RoleAddr]], int]:
        # get master address
        if not master_addr:
            return None
        # prepare request to master
        url = "http://" + master_addr + "/rtp_llm/schedule"
        payload = {
            "model": "engine_service",
            "block_cache_keys": block_cache_keys,
            "seq_len": seq_len,
            "debug": debug,
            "request_priority": request_priority,
            "request_id": request_id,
        }
        if generate_timeout != -1:
            payload["generate_timeout"] = generate_timeout
        headers = {"Content-Type": "application/json"}

        # connect to master using long connection
        try:
            session = await self._get_session()
            async with session.post(
                url, data=json.dumps(payload), headers=headers
            ) as response:
                if response.status != 200:
                    route_logger.error(
                        f"Failed to get master response from {master_addr}, http status: {response.status}"
                    )
                    return None
                result = await response.json()
        except Exception as e:
            route_logger.error(f"Failed to connect to master at {master_addr}: {e}")
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
