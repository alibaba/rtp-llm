"""K3.1 evidence-port checks in addition to the shared smoke configuration."""

from rtp_llm.test.smoke.kimi_k3.pd_config import validate_config as shared_validate

REQUEST_STOP_TIMEOUT_S = 60
SERVICE_STOP_TIMEOUT_S = 330
GPU_CLEANUP_TIMEOUT_S = 60
PEER_CLEANUP_TIMEOUT_S = (
    REQUEST_STOP_TIMEOUT_S + SERVICE_STOP_TIMEOUT_S + GPU_CLEANUP_TIMEOUT_S + 60
)


def validate_config(config, *, local_role):
    shared_validate(config, local_role=local_role)
    if config.get("smoke_profile") != "k31":
        raise ValueError("The CI adapter requires the K3.1 smoke profile")
    from rtp_llm.models.kimi_k3.nvfp4_config import parse_nvfp4_flag

    for key in ("kda_nvfp4", "dsa_nvfp4"):
        config[key] = parse_nvfp4_flag(key.upper(), config.get(key), default=True)
    for role in ("prefill", "decode"):
        host = config[role]
        port = host.get("evidence_port")
        if type(port) is not int or not 1024 <= port <= 65535:
            raise ValueError("Invalid execution evidence port")
        if host["port"] <= port < host["port"] + 72:
            raise ValueError("Execution evidence port overlaps the service port block")
    return config
