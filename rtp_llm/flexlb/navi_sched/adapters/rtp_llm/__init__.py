from importlib import import_module

__all__ = [
    "NaviSchedAdapterConfig",
    "NaviSchedClient",
    "FlexlbInfo",
    "create_navi_sched_client",
    "apply_frontend_request_overrides",
    "SwitchableMasterClient",
    "create_switchable_master_client",
    "AutoDetectMasterClient",
    "create_auto_detect_master_client",
    "get_master_type",
    "get_schedule_timeout_ms",
    "split_address",
]

_EXPORTS = {
    "NaviSchedAdapterConfig": (
        ".NaviSchedAdapterConfig",
        "NaviSchedAdapterConfig",
    ),
    "NaviSchedClient": (".NaviSchedClient", "NaviSchedClient"),
    "FlexlbInfo": (".flexlb_info", "FlexlbInfo"),
    "create_navi_sched_client": (
        ".NaviSchedClient",
        "create_navi_sched_client",
    ),
    "apply_frontend_request_overrides": (
        ".NaviSchedClient",
        "apply_frontend_request_overrides",
    ),
    "SwitchableMasterClient": (
        ".SwitchableMasterClient",
        "SwitchableMasterClient",
    ),
    "create_switchable_master_client": (
        ".SwitchableMasterClient",
        "create_switchable_master_client",
    ),
    "AutoDetectMasterClient": (
        ".SwitchableMasterClient",
        "AutoDetectMasterClient",
    ),
    "create_auto_detect_master_client": (
        ".SwitchableMasterClient",
        "create_auto_detect_master_client",
    ),
    "get_master_type": (".RtpLLMAdapterUtils", "get_master_type"),
    "get_schedule_timeout_ms": (
        ".RtpLLMAdapterUtils",
        "get_schedule_timeout_ms",
    ),
    "split_address": (".RtpLLMAdapterUtils", "split_address"),
}


def __getattr__(name):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
