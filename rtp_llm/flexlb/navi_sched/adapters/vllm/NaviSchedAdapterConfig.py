from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class NaviSchedAdapterConfig:
    master_biz_name: str
    target_biz_name: str
    client_biz_name: str = "vllm_router"
    master_addresses: tuple[str, ...] = ()
    master_vip_domain: str = ""
    master_vip_jmenv: str = "jmenv.tbsite.net"
    master_vip_grpc_port_diff: int = 2
    timeout_ms: int = 1000
    block_size: int = 16
    route_worker_count: int = 64
    replay_sample_interval: int = 0

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        block_size: int | None = None,
    ) -> NaviSchedAdapterConfig:
        values = environ if environ is not None else os.environ
        addresses = tuple(
            address.strip()
            for address in values.get("VLLM_NAVI_SCHED_MASTER_ADDRESSES", "").split(",")
            if address.strip()
        )
        config = cls(
            master_biz_name=values.get("VLLM_NAVI_SCHED_MASTER_BIZ_NAME", "").strip(),
            target_biz_name=values.get("VLLM_NAVI_SCHED_TARGET_BIZ_NAME", "").strip(),
            client_biz_name=values.get(
                "VLLM_NAVI_SCHED_CLIENT_BIZ_NAME", "vllm_router"
            ).strip(),
            master_addresses=addresses,
            master_vip_domain=values.get(
                "VLLM_NAVI_SCHED_MASTER_VIP_DOMAIN", ""
            ).strip(),
            master_vip_jmenv=values.get(
                "VLLM_NAVI_SCHED_MASTER_VIP_JMENV", "jmenv.tbsite.net"
            ).strip(),
            master_vip_grpc_port_diff=int(
                values.get("VLLM_NAVI_SCHED_MASTER_GRPC_PORT_DIFF", "2")
            ),
            timeout_ms=int(values.get("VLLM_NAVI_SCHED_TIMEOUT_MS", "1000")),
            block_size=(
                int(values.get("VLLM_NAVI_SCHED_BLOCK_SIZE", "16"))
                if block_size is None
                else int(block_size)
            ),
            route_worker_count=int(
                values.get("VLLM_NAVI_SCHED_ROUTE_WORKER_COUNT", "64")
            ),
            replay_sample_interval=int(
                values.get("VLLM_NAVI_SCHED_REPLAY_SAMPLE_INTERVAL", "0")
            ),
        )
        config.validate_client()
        return config

    def validate_client(self) -> None:
        if not self.target_biz_name:
            raise ValueError("VLLM_NAVI_SCHED_TARGET_BIZ_NAME is required")
        if not self.client_biz_name:
            raise ValueError("VLLM_NAVI_SCHED_CLIENT_BIZ_NAME is required")
        if not self.master_biz_name:
            raise ValueError("VLLM_NAVI_SCHED_MASTER_BIZ_NAME is required")
        if bool(self.master_addresses) == bool(self.master_vip_domain):
            raise ValueError("exactly one NaviSched Master discovery mode is required")
        if not -65535 <= self.master_vip_grpc_port_diff <= 65535:
            raise ValueError(
                "VLLM_NAVI_SCHED_MASTER_GRPC_PORT_DIFF must be between "
                "-65535 and 65535"
            )
        if self.timeout_ms <= 0:
            raise ValueError("VLLM_NAVI_SCHED_TIMEOUT_MS must be positive")
        if self.block_size <= 0:
            raise ValueError("VLLM_NAVI_SCHED_BLOCK_SIZE must be positive")
        if self.route_worker_count <= 0:
            raise ValueError("VLLM_NAVI_SCHED_ROUTE_WORKER_COUNT must be positive")
        if self.replay_sample_interval < 0:
            raise ValueError(
                "VLLM_NAVI_SCHED_REPLAY_SAMPLE_INTERVAL must be nonnegative"
            )
