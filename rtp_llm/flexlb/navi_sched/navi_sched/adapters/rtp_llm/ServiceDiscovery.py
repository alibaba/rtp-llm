import json
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

MODEL_SERVICE_CONFIG_ENV = "MODEL_SERVICE_CONFIG"
VIPSERVER_JMENV_ENV = "com.alibaba.vipserver.jmenv"
DEFAULT_VIPSERVER_JMENV = "jmenv.tbsite.net"
DEFAULT_MASTER_BIZ_NAME = "rtp_llm_navi_sched_master"
NAVI_SCHED_BIZ_NAME_ENV = "RTP_LLM_NAVI_SCHED_BIZ_NAME"


@dataclass(frozen=True)
class ServiceEndpoint:
    address: str
    protocol: str = "http"

    @classmethod
    def from_value(cls, value, name: str) -> "ServiceEndpoint":
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a JSON object")
        address = str(value.get("address", "")).strip()
        if not address:
            raise ValueError(f"{name}.address is required")
        protocol = str(value.get("protocol", "http")).strip().lower()
        return cls(address=address, protocol=protocol or "http")


@dataclass(frozen=True)
class RtpLLMServiceRoute:
    service_id: str
    master_endpoint: Optional[ServiceEndpoint]
    prefill_endpoint: Optional[ServiceEndpoint]

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str], required: bool = True
    ) -> Optional["RtpLLMServiceRoute"]:
        raw = environ.get(MODEL_SERVICE_CONFIG_ENV, "").strip()
        if not raw:
            if required:
                raise ValueError(f"{MODEL_SERVICE_CONFIG_ENV} is required")
            return None
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{MODEL_SERVICE_CONFIG_ENV} must be valid JSON: {error.msg}"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(f"{MODEL_SERVICE_CONFIG_ENV} must be a JSON object")

        master_value = value.get("master_endpoint")
        master_endpoint = (
            ServiceEndpoint.from_value(master_value, "master_endpoint")
            if master_value is not None
            else None
        )
        prefill_endpoints = []
        role_endpoints = value.get("role_endpoints", [])
        if not isinstance(role_endpoints, list):
            raise ValueError("role_endpoints must be a JSON array")
        for index, group in enumerate(role_endpoints):
            if not isinstance(group, dict):
                raise ValueError(f"role_endpoints[{index}] must be a JSON object")
            endpoint = group.get("prefill_endpoint")
            if endpoint is not None:
                prefill_endpoints.append(
                    ServiceEndpoint.from_value(
                        endpoint, f"role_endpoints[{index}].prefill_endpoint"
                    )
                )
        unique_prefill = set(prefill_endpoints)
        if len(unique_prefill) > 1:
            raise ValueError(
                "NaviSched currently requires all prefill endpoints to use the "
                "same address and protocol"
            )
        if value.get("use_local", False):
            raise ValueError("only VIP service discovery is supported")
        return cls(
            service_id=str(value.get("service_id", "")).strip(),
            master_endpoint=master_endpoint,
            prefill_endpoint=(next(iter(unique_prefill)) if unique_prefill else None),
        )


def master_grpc_port_diff(endpoint: ServiceEndpoint) -> int:
    return 0 if endpoint.protocol == "grpc" else 2


def agent_grpc_port_diff(endpoint: ServiceEndpoint) -> int:
    return 2 if endpoint.protocol == "grpc" else 3


def make_discovery_runtime(
    endpoint: ServiceEndpoint,
    grpc_port_diff: int,
    environ: Mapping[str, str],
):
    return {
        "vip_subscription": {
            "jmenv_domain": environ.get(
                VIPSERVER_JMENV_ENV, DEFAULT_VIPSERVER_JMENV
            ),
            "services": [endpoint.address],
            "grpc_port_diff": grpc_port_diff,
        }
    }


def derive_prefill_service(
    environ: Mapping[str, str],
) -> Tuple[RtpLLMServiceRoute, ServiceEndpoint, str]:
    route = RtpLLMServiceRoute.from_env(environ)
    if route.prefill_endpoint is None:
        raise ValueError("MODEL_SERVICE_CONFIG has no prefill endpoint")
    endpoint = route.prefill_endpoint
    return route, endpoint, endpoint.address


def prefill_biz_name(
    environ: Mapping[str, str], endpoint: ServiceEndpoint, configured: str = ""
) -> str:
    return (
        str(configured).strip()
        or environ.get(NAVI_SCHED_BIZ_NAME_ENV, "").strip()
        or environ.get("BIZ_NAME", "").strip()
        or endpoint.address
    )
