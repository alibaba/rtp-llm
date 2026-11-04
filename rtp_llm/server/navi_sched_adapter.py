import importlib
import os
from typing import Mapping, Optional

MASTER_TYPE_ENV = "RTP_LLM_MASTER_TYPE"
FLEXLB_MASTER_TYPE = "flexlb"
NAVI_SCHED_MASTER_TYPE = "navi_sched"
NAVI_SCHED_CONFIG_JSON_ENV = "RTP_LLM_NAVI_SCHED_CONFIG_JSON"


def get_master_type(environ: Optional[Mapping[str, str]] = None) -> str:
    values = environ if environ is not None else os.environ
    master_type = values.get(MASTER_TYPE_ENV, "").strip().lower()
    if not master_type:
        return FLEXLB_MASTER_TYPE
    if master_type not in (FLEXLB_MASTER_TYPE, NAVI_SCHED_MASTER_TYPE):
        raise ValueError(
            f"{MASTER_TYPE_ENV} must be {FLEXLB_MASTER_TYPE} or "
            f"{NAVI_SCHED_MASTER_TYPE}, got {master_type}"
        )
    return master_type


def _load_adapter():
    return importlib.import_module("navi_sched.adapters.rtp_llm")


def create_navi_sched_client(host_service):
    if not navi_sched_enabled():
        return None
    return _load_adapter().create_navi_sched_client(host_service)


def get_navi_sched_backend_roles(default_roles, navi_sched_client):
    config = getattr(navi_sched_client, "config", None)
    get_target_role = getattr(config, "get_target_role", None)
    if get_target_role is None:
        return default_roles
    return [get_target_role()]


def apply_navi_sched_request_overrides(input):
    if not os.environ.get(NAVI_SCHED_CONFIG_JSON_ENV, "").strip():
        return
    _load_adapter().apply_frontend_request_overrides(input)


def create_auto_detect_master_client(flexlb_client, host_service):
    try:
        adapter = _load_adapter()
    except ImportError:
        return flexlb_client
    return adapter.create_auto_detect_master_client(flexlb_client, host_service)


def navi_sched_enabled():
    master_type = os.environ.get(MASTER_TYPE_ENV, "").strip()
    if master_type:
        return get_master_type() == NAVI_SCHED_MASTER_TYPE
    return bool(os.environ.get(NAVI_SCHED_CONFIG_JSON_ENV, "").strip())
