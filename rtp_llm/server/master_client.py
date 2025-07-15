import json
import logging
from typing import List, Optional, Tuple

import aiohttp
from aiohttp import ClientTimeout

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import RoleAddr, RoleType
from rtp_llm.server.worker_status import ScheduleMeta


class MasterClient:
    async def get_backend_role_addrs(
        self,
        master_addr: Optional[str],
        block_cache_keys: list[int],
        seq_len: int,
        debug: bool,
    ) -> Tuple[Optional[List[RoleAddr]], int]:
        inter_request_id = -1
        # get master address
        if not master_addr:
            return None, inter_request_id

        # prepare request to master
        url = "http://" + master_addr + "/rtp_llm/schedule"
        payload = {
            "model": "engine_service",
            "block_cache_keys": block_cache_keys,
            "seq_len": seq_len,
            "debug": debug,
        }
        headers = {"Content-Type": "application/json"}

        # connect to master
        timeout = ClientTimeout(total=0.5)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    url, data=json.dumps(payload), headers=headers
                ) as response:
                    if response.status != 200:
                        return None, inter_request_id
                    result = await response.json()
        except aiohttp.ClientError as e:
            logging.error(f"Failed to connect to master at {master_addr}: {e}")
            return None, inter_request_id

        # check response
        schedule_meta = ScheduleMeta.model_validate(result)
        if schedule_meta.code != 200:
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
