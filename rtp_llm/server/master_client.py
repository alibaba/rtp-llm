import json
import logging
import os
import time
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
        self.timeout_threshold_ms = float(os.environ.get('MASTER_CLIENT_TIMEOUT_MS', '10'))

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
        request_priority: int = 100,
    ) -> Tuple[Optional[List[RoleAddr]], int]:

        start_time = time.time()
        inter_request_id = -1
        # get master address
        if not master_addr:
            # 计算并记录RT
            elapsed_time = (time.time() - start_time) * 1000
            if elapsed_time > self.timeout_threshold_ms:
                route_logger.warning(f"get_backend_role_addrs RT exceeded {self.timeout_threshold_ms}ms: {elapsed_time:.2f}ms (no master_addr)")
            return None, inter_request_id

        # prepare request to master
        url = "http://" + master_addr + "/rtp_llm/schedule"
        if generate_timeout != -1:
            payload = {
                "model": "engine_service",
                "block_cache_keys": block_cache_keys,
                "seq_len": seq_len,
                "debug": debug,
                "generate_timeout": generate_timeout,
                "request_priority": request_priority,
            }
        else:
            payload = {
                "model": "engine_service",
                "block_cache_keys": block_cache_keys,
                "seq_len": seq_len,
                "debug": debug,
                "request_priority": request_priority,
            }
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
                    # 计算并记录RT
                    elapsed_time = (time.time() - start_time) * 1000
                    if elapsed_time > self.timeout_threshold_ms:
                        route_logger.warning(f"get_backend_role_addrs RT exceeded {self.timeout_threshold_ms}ms: {elapsed_time:.2f}ms (http status: {response.status})")
                    return None, inter_request_id
                result = await response.json()
        except Exception as e:
            route_logger.error(f"query master[{master_addr}] failed: {type(e).__name__}: {e}")
            # 计算并记录RT
            elapsed_time = (time.time() - start_time) * 1000
            if elapsed_time > self.timeout_threshold_ms:
                route_logger.warning(f"get_backend_role_addrs RT exceeded {self.timeout_threshold_ms}ms: {elapsed_time:.2f}ms (exception occurred)")
            return None, inter_request_id

        # 计算并记录RT
        elapsed_time = (time.time() - start_time) * 1000
        if elapsed_time > self.timeout_threshold_ms:
            route_logger.warning(f"get_backend_role_addrs RT exceeded {self.timeout_threshold_ms}ms: {elapsed_time:.2f}ms (success)")

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

        return role_addrs, schedule_meta.inter_request_id
