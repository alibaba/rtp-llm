from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SGLangConfig:
    target_biz: str
    master_biz: str
    runtime: dict = field(default_factory=dict)
    client_biz: str = "sglang_client"
    schedule_timeout_ms: int = 1000
    request_timeout_ms: int = 300000
    max_payload_bytes: int = 16 * 1024 * 1024
    max_inflight_requests: int = 1024
    max_buffered_responses: int = 64
    agent_port: int = 0
    agent_grpc_port: int = 0
    server_ip: str = ""

    @classmethod
    def from_env(cls):
        prefix = "SGLANG_NAVI_SCHED_"
        values = {
            "target_biz": os.environ.get(prefix + "TARGET_BIZ", ""),
            "master_biz": os.environ.get(prefix + "MASTER_BIZ", ""),
            "runtime": json.loads(os.environ.get(prefix + "RUNTIME_CONFIG", "{}")),
        }
        for name in (
            "schedule_timeout_ms",
            "request_timeout_ms",
            "max_payload_bytes",
            "max_inflight_requests",
            "max_buffered_responses",
            "agent_port",
            "agent_grpc_port",
        ):
            if prefix + name.upper() in os.environ:
                values[name] = int(os.environ[prefix + name.upper()])
        for name in ("client_biz", "server_ip"):
            if prefix + name.upper() in os.environ:
                values[name] = os.environ[prefix + name.upper()]
        result = cls(**values)
        result.validate()
        return result

    def validate(self):
        if not self.target_biz or not self.master_biz or not self.client_biz:
            raise ValueError(
                "SGLang NaviSched requires target_biz, master_biz and client_biz"
            )
        if not isinstance(self.runtime, dict):
            raise ValueError("runtime must be a NaviSched runtime configuration object")
        for name in (
            "schedule_timeout_ms",
            "request_timeout_ms",
            "max_payload_bytes",
            "max_inflight_requests",
            "max_buffered_responses",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name + " must be a positive integer")
        if not 0 <= self.agent_port <= 65535:
            raise ValueError("agent_port must be between 0 and 65535")
        if not 0 <= self.agent_grpc_port <= 65535:
            raise ValueError("agent_grpc_port must be between 0 and 65535")
