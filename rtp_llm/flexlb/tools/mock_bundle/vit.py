"""Run one CPU Mock ViT per Pod without loading the frontend/Torch runtime."""

import json
import os
import shlex
from pathlib import Path

from whale_frontend_fixture import prepare_assets


def command(env):
    if env.get("RTP_LLM_MOCK_BUNDLE") != "1":
        raise ValueError("RTP_LLM_MOCK_BUNDLE=1 is required")
    if env.get("ROLE_TYPE") != "VIT":
        raise ValueError("ROLE_TYPE=VIT is required")
    kmonitor = env.get("MOCK_KMONITOR_ENABLED", "true").lower()
    if kmonitor not in ("true", "false", "1", "0", "yes", "no"):
        raise ValueError("invalid MOCK_KMONITOR_ENABLED")
    kmonitor = "true" if kmonitor in ("true", "1", "yes") else "false"
    port = int(env.get("START_PORT", "22160"))
    if not 0 < port < 65535:
        raise ValueError("invalid ViT health port")
    source = env.get("CHECKPOINT_PATH") or env.get("TOKENIZER_PATH")
    if not source:
        raise ValueError("ViT requires real model metadata")
    runtime = Path(env.get("HIPPO_PROC_WORKDIR", "/tmp")) / "mock-vit-assets"
    model = prepare_assets(source, runtime)
    from standalone_mock import command as cluster_command

    if env.get("MOCK_DEPLOYMENT_MODE", "per_engine") != "per_engine":
        raise ValueError("VIT role launcher requires MOCK_DEPLOYMENT_MODE=per_engine")
    return cluster_command({**env, "MOCK_DEPLOYMENT_MODE": "per_engine", "START_PORT": str(port),
        "MOCK_VIT_MODEL_PATH": str(model), "MOCK_BUNDLE_RUN_DIR": str(runtime),
        "MOCK_JAVA_OPTS": env.get("MOCK_VIT_JAVA_OPTS", "-Xms64m -Xmx512m -Djava.awt.headless=true"),
        "MOCK_KMONITOR_ENABLED": kmonitor})


if __name__ == "__main__":
    args = command(os.environ)
    os.execv(args[0], args)
