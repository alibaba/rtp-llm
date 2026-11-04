import os
from dataclasses import dataclass
from typing import Mapping, Optional

from rtp_llm.config.generate_config import RoleType

from .RtpLLMAdapterUtils import (
    NAVI_SCHED_MASTER_TYPE,
    get_master_type,
)
from .ServiceDiscovery import DEFAULT_MASTER_BIZ_NAME, RtpLLMServiceRoute
from .RtpLLMUnifiedConfig import load_navi_sched_config

NAVI_SCHED_MASTER_BIZ_NAME_ENV = "RTP_LLM_NAVI_SCHED_MASTER_BIZ_NAME"
NAVI_SCHED_BIZ_NAME_ENV = "RTP_LLM_NAVI_SCHED_BIZ_NAME"
NAVI_SCHED_TARGET_ROLE_ENV = "RTP_LLM_NAVI_SCHED_TARGET_ROLE"
NAVI_SCHED_MASTER_ADDRESS_ENV = "RTP_LLM_NAVI_SCHED_MASTER_ADDRESS"
NAVI_SCHED_REPLAY_SAMPLE_INTERVAL_ENV = (
    "RTP_LLM_NAVI_SCHED_REPLAY_SAMPLE_INTERVAL"
)
NAVI_SCHED_REPLAY_DIR_ENV = "RTP_LLM_NAVI_SCHED_REPLAY_DIR"


@dataclass(frozen=True)
class NaviSchedAdapterConfig:
    enabled: bool = False
    master_biz_name: str = ""
    biz_name: str = ""
    target_role: str = "PREFILL"
    master_address: str = ""
    replay_sample_interval: int = 0
    replay_dir: str = ""
    log_path: str = ""
    force_max_new_tokens: int = 0

    @classmethod
    def from_env(
        cls, environ: Optional[Mapping[str, str]] = None
    ) -> "NaviSchedAdapterConfig":
        values = environ if environ is not None else os.environ
        unified_config = load_navi_sched_config(values)
        client_config = unified_config.get("client", {})
        initial_route_mode = get_master_type(values)
        enabled = (
            initial_route_mode == NAVI_SCHED_MASTER_TYPE
            or bool(unified_config)
        )
        master_biz_name = values.get(
            NAVI_SCHED_MASTER_BIZ_NAME_ENV, ""
        ).strip() or DEFAULT_MASTER_BIZ_NAME
        biz_name = str(client_config.get(
            "biz_name",
            values.get(NAVI_SCHED_BIZ_NAME_ENV, "")
            or values.get("BIZ_NAME", ""),
        )).strip()
        if not master_biz_name or not biz_name:
            route = RtpLLMServiceRoute.from_env(values, required=False)
            if route is not None:
                if not biz_name and route.prefill_endpoint is not None:
                    biz_name = route.prefill_endpoint.address
        return cls(
            enabled=enabled,
            master_biz_name=master_biz_name,
            biz_name=biz_name,
            target_role=str(client_config.get(
                "target_role",
                values.get(NAVI_SCHED_TARGET_ROLE_ENV, "PREFILL"),
            )).strip(),
            master_address=values.get(
                NAVI_SCHED_MASTER_ADDRESS_ENV, ""
            ).strip(),
            replay_sample_interval=int(client_config.get(
                "replay_sample_interval",
                values.get(NAVI_SCHED_REPLAY_SAMPLE_INTERVAL_ENV, "0") or 0,
            )),
            replay_dir=str(client_config.get(
                "replay_dir", values.get(NAVI_SCHED_REPLAY_DIR_ENV, "")
            )).strip(),
            log_path=str(unified_config.get("log_path", "")).strip(),
            force_max_new_tokens=int(
                client_config.get("force_max_new_tokens", 0) or 0
            ),
        )

    def validate_client(self) -> None:
        if not self.biz_name:
            raise ValueError("RTP_LLM_NAVI_SCHED_BIZ_NAME is required")
        if self.replay_sample_interval < 0:
            raise ValueError(
                "RTP_LLM_NAVI_SCHED_REPLAY_SAMPLE_INTERVAL must be nonnegative"
            )
        if self.replay_sample_interval and not self.replay_dir:
            raise ValueError(
                "RTP_LLM_NAVI_SCHED_REPLAY_DIR is required when replay sampling is enabled"
            )
        if self.force_max_new_tokens < 0:
            raise ValueError("client.force_max_new_tokens must be nonnegative")
        self.get_target_role()

    def validate_switching(self) -> None:
        self.validate_client()
        if not self.master_address:
            raise ValueError("RTP_LLM_NAVI_SCHED_MASTER_ADDRESS is required")

    def get_target_role(self) -> RoleType:
        try:
            return getattr(RoleType, self.target_role.upper())
        except AttributeError as error:
            raise ValueError(
                f"invalid RTP_LLM_NAVI_SCHED_TARGET_ROLE: {self.target_role}"
            ) from error
