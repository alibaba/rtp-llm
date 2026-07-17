from __future__ import annotations

import argparse
import os
import socket
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _default_kvcm_url() -> str:
    explicit = os.environ.get("KVCM_URL", "")
    if explicit:
        return explicit
    virtual_service_id = os.environ.get("KVCM_VSERVICE_ID", "")
    if virtual_service_id:
        return f"spectrum://{virtual_service_id}:6382"
    return ""


def _local_host_identity() -> str:
    try:
        address = socket.gethostbyname(socket.gethostname())
    except OSError:
        address = "127.0.0.1"
    return f"{address}:8088"


@dataclass(frozen=True)
class SubscriberConfig:
    rtp_endpoints: tuple[str, ...]
    rtp_rpc_timeout_s: float
    poll_interval_s: float
    deletion_confirmations: int
    engine_failure_threshold: int
    full_refresh_interval_s: float
    kvcm_url: str
    kvcm_request_timeout_s: float
    kvcm_heartbeat_interval_s: float
    kvcm_report_batch_size: int
    instance_id: str
    instance_group: str
    host_ip_port: str
    storage_type: str
    medium: str
    reset_on_start: bool
    log_level: str

    def validate(self) -> None:
        if not self.rtp_endpoints or any(not endpoint for endpoint in self.rtp_endpoints):
            raise ValueError("rtp_endpoints must contain at least one endpoint")
        if not self.kvcm_url:
            raise ValueError("KVCM_URL or KVCM_VSERVICE_ID must be configured")
        if not self.instance_id:
            raise ValueError(
                "KVCM_INSTANCE_ID or SPECTRUM_DEPLOYMENT_NAME must be configured"
            )
        if self.rtp_rpc_timeout_s <= 0:
            raise ValueError("rtp_rpc_timeout_s must be > 0")
        if self.poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be > 0")
        if self.deletion_confirmations < 1:
            raise ValueError("deletion_confirmations must be >= 1")
        if self.engine_failure_threshold < 1:
            raise ValueError("engine_failure_threshold must be >= 1")
        if self.full_refresh_interval_s <= 0:
            raise ValueError("full_refresh_interval_s must be > 0")
        if self.kvcm_request_timeout_s <= 0:
            raise ValueError("kvcm_request_timeout_s must be > 0")
        if self.kvcm_heartbeat_interval_s <= 0:
            raise ValueError("kvcm_heartbeat_interval_s must be > 0")
        if self.kvcm_report_batch_size < 1:
            raise ValueError("kvcm_report_batch_size must be >= 1")
        if not self.host_ip_port:
            raise ValueError("host_ip_port must not be empty")
        if not self.storage_type:
            raise ValueError("storage_type must not be empty")
        if not self.medium:
            raise ValueError("medium must not be empty")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RTP-LLM full-cache subscriber for Tair KVCM"
    )
    parser.add_argument(
        "--rtp-endpoints",
        default=os.environ.get("RTP_LLM_CACHE_SUBSCRIBER_ENDPOINTS", "127.0.0.1:8089"),
        help="Comma-separated RTP GetCacheStatus gRPC endpoints",
    )
    parser.add_argument("--rtp-rpc-timeout-s", type=float, default=1.0)
    parser.add_argument("--poll-interval-s", type=float, default=1.0)
    parser.add_argument("--deletion-confirmations", type=int, default=2)
    parser.add_argument("--engine-failure-threshold", type=int, default=3)
    parser.add_argument("--full-refresh-interval-s", type=float, default=300.0)
    parser.add_argument("--kvcm-url", default=_default_kvcm_url())
    parser.add_argument("--kvcm-request-timeout-s", type=float, default=5.0)
    parser.add_argument("--kvcm-heartbeat-interval-s", type=float, default=1.0)
    parser.add_argument("--kvcm-report-batch-size", type=int, default=1000)
    parser.add_argument(
        "--instance-id",
        default=os.environ.get(
            "KVCM_INSTANCE_ID", os.environ.get("SPECTRUM_DEPLOYMENT_NAME", "")
        ),
    )
    parser.add_argument(
        "--instance-group", default=os.environ.get("KVCM_INSTANCE_GROUP", "")
    )
    parser.add_argument(
        "--host-ip-port",
        default=os.environ.get("KVCM_HOST_IP_PORT", _local_host_identity()),
    )
    parser.add_argument(
        "--storage-type",
        default=os.environ.get("KVCM_STORAGE_TYPE", "ST_EVENT_REPORT"),
    )
    parser.add_argument("--medium", default=os.environ.get("KVCM_MEDIUM", "hbm"))
    parser.add_argument(
        "--reset-on-start",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("KVCM_RESET_ON_START", True),
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=os.environ.get("LOG_LEVEL", "INFO").upper(),
    )
    return parser


def config_from_args(args: argparse.Namespace) -> SubscriberConfig:
    endpoints = tuple(
        endpoint.strip() for endpoint in args.rtp_endpoints.split(",") if endpoint.strip()
    )
    config = SubscriberConfig(
        rtp_endpoints=endpoints,
        rtp_rpc_timeout_s=args.rtp_rpc_timeout_s,
        poll_interval_s=args.poll_interval_s,
        deletion_confirmations=args.deletion_confirmations,
        engine_failure_threshold=args.engine_failure_threshold,
        full_refresh_interval_s=args.full_refresh_interval_s,
        kvcm_url=args.kvcm_url,
        kvcm_request_timeout_s=args.kvcm_request_timeout_s,
        kvcm_heartbeat_interval_s=args.kvcm_heartbeat_interval_s,
        kvcm_report_batch_size=args.kvcm_report_batch_size,
        instance_id=args.instance_id,
        instance_group=args.instance_group,
        host_ip_port=args.host_ip_port,
        storage_type=args.storage_type,
        medium=args.medium,
        reset_on_start=args.reset_on_start,
        log_level=args.log_level,
    )
    config.validate()
    return config
