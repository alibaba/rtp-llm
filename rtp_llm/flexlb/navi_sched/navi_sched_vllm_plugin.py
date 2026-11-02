import os


def register_vllm_plugin() -> None:
    enabled = os.environ.get("VLLM_NAVI_SCHED_ENABLE", "1").strip()
    if enabled not in ("0", "1"):
        raise ValueError("VLLM_NAVI_SCHED_ENABLE must be 0 or 1")
    if enabled == "0":
        return

    from navi_sched.adapters.vllm.NaviSchedVLLMPlugin import (
        register_enabled_vllm_plugin,
    )

    register_enabled_vllm_plugin()
