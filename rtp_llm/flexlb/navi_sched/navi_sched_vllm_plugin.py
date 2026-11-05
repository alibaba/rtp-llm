import os


def register_vllm_plugin() -> None:
    enabled = os.environ.get("VLLM_NAVI_SCHED_ENABLE", "1").strip()
    if enabled not in ("0", "1"):
        raise ValueError("VLLM_NAVI_SCHED_ENABLE must be 0 or 1")
    if enabled == "0":
        return

    # DashLLM's Decode engine is fed by its own P/D orchestration. Routing it
    # through the Prefill Agent would send Decode requests to the wrong role.
    if (
        os.environ.get("DS_ENABLE_DSERV") == "1"
        and os.environ.get("TPU_RAIDEN_JOB_NAME", "").strip().lower() == "decode"
    ):
        return

    from navi_sched.adapters.vllm.NaviSchedVLLMPlugin import (
        register_enabled_vllm_plugin,
    )

    register_enabled_vllm_plugin()
