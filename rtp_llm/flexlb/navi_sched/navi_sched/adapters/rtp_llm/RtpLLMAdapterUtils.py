import os
from typing import Mapping, Tuple

MASTER_TYPE_ENV = "RTP_LLM_MASTER_TYPE"
FLEXLB_MASTER_TYPE = "flexlb"
NAVI_SCHED_MASTER_TYPE = "navi_sched"
VIPSERVER_JMENV_ENV = "com.alibaba.vipserver.jmenv"
DEFAULT_VIPSERVER_JMENV = "jmenv.tbsite.net"
LOG_PATH_ENV = "LOG_PATH"
KMON_SERVICE_NAME_ENV = "kmonitorServiceName"
DEFAULT_KMON_SERVICE_NAME = "navi_sched"


def get_master_type(environ: Mapping[str, str]) -> str:
    master_type = environ.get(MASTER_TYPE_ENV, "").strip().lower()
    if not master_type:
        return FLEXLB_MASTER_TYPE
    if master_type not in (FLEXLB_MASTER_TYPE, NAVI_SCHED_MASTER_TYPE):
        raise ValueError(
            f"{MASTER_TYPE_ENV} must be {FLEXLB_MASTER_TYPE} or "
            f"{NAVI_SCHED_MASTER_TYPE}, got {master_type}"
        )
    return master_type


def split_address(address: str) -> Tuple[str, int]:
    value = address.strip()
    if value.startswith("["):
        bracket = value.find("]")
        if bracket <= 1 or value[bracket + 1:bracket + 2] != ":":
            raise ValueError(f"invalid server address: {address}")
        return value[1:bracket], int(value[bracket + 2:])
    ip, separator, port = value.rpartition(":")
    if not separator or not ip or not port:
        raise ValueError(f"invalid server address: {address}")
    return ip, int(port)


def get_schedule_timeout_ms(generate_config) -> int:
    for name in ("ttft_timeout_ms", "timeout_ms"):
        value = getattr(generate_config, name, None)
        if value is not None and int(value) > 0:
            return int(value)
    return 1000


def make_runtime_options(name: str, configured_log_path: str = ""):
    log_dir = configured_log_path or os.environ.get(LOG_PATH_ENV, "logs")
    runtime_dir = os.path.join(log_dir, "navi_sched")
    os.makedirs(runtime_dir, exist_ok=True)
    return {
        "log_file_name": os.path.join(log_dir, "navi_sched.log"),
        "gig": {
            "log_prefix": name,
            "snapshot_log_dir": runtime_dir,
        },
        "snapshot_dir": runtime_dir,
    }


def make_kmon_options(role: str, environ: Mapping[str, str]):
    service_name = environ.get(KMON_SERVICE_NAME_ENV, "").strip()
    tags = {"role": role}
    deployment_name = environ.get("DEPLOYMENT_NAME", "")
    if deployment_name:
        tags["deployment_name"] = deployment_name
    return {
        "enable_kmon": True,
        "kmon_service_name": service_name or DEFAULT_KMON_SERVICE_NAME,
        "kmon_metrics_path": "",
        "kmon_tags": tags,
    }
