"""Model-specific controls shared by startup kernel and gRPC warmup."""

import os

_AUTO_WARMUP_MODELS = frozenset(("deepseek_v4", "kimi_k3"))


def startup_real_warmup_flag_env(model_type: str) -> str:
    if model_type == "kimi_k3":
        return "KIMI_K3_STARTUP_REAL_WARMUP"
    return "DSV4_STARTUP_REAL_WARMUP"


def startup_real_warmup_timeout_env(model_type: str) -> str:
    if model_type == "kimi_k3":
        return "KIMI_K3_STARTUP_REAL_WARMUP_TIMEOUT_S"
    return "DSV4_STARTUP_REAL_WARMUP_TIMEOUT_S"


def startup_real_warmup_auto_enabled(model_type: str) -> bool:
    return model_type in _AUTO_WARMUP_MODELS


def startup_real_warmup_enabled(model_type: str) -> bool:
    """Resolve the one model-specific switch used by every startup warmup."""

    flag = os.environ.get(startup_real_warmup_flag_env(model_type), "auto")
    normalized = flag.strip().lower()
    if normalized in ("0", "false", "off", "no"):
        return False
    if normalized in ("1", "true", "on", "yes", "force"):
        return True
    return startup_real_warmup_auto_enabled(model_type)


__all__ = [
    "startup_real_warmup_auto_enabled",
    "startup_real_warmup_enabled",
    "startup_real_warmup_flag_env",
    "startup_real_warmup_timeout_env",
]
