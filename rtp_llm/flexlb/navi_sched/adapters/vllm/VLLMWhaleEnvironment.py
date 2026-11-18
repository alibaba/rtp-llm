"""Emit shell exports for the NaviSched-enabled Whale worker process."""

import os
import shlex
import socket
import sys


def build_environment(environ, start_port, hippo_app, resolve_ip):
    biz_name = (
        environ.get("VLLM_NAVI_SCHED_BIZ_NAME")
        or environ.get("VLLM_NAVI_SCHED_TARGET_BIZ_NAME")
        or hippo_app
    )
    return {
        "VLLM_NAVI_SCHED_SERVER_IP": environ.get("VLLM_NAVI_SCHED_SERVER_IP")
        or resolve_ip(),
        "VLLM_NAVI_SCHED_SERVER_PORT": environ.get("VLLM_NAVI_SCHED_SERVER_PORT")
        or str(start_port),
        "VLLM_NAVI_SCHED_AGENT_GIG_PORT": environ.get("VLLM_NAVI_SCHED_AGENT_GIG_PORT")
        or str(int(start_port) + 4),
        "VLLM_NAVI_SCHED_MASTER_BIZ_NAME": environ.get(
            "VLLM_NAVI_SCHED_MASTER_BIZ_NAME"
        )
        or "vllm_navi_sched_master",
        "VLLM_NAVI_SCHED_BIZ_NAME": biz_name,
        "VLLM_NAVI_SCHED_TARGET_BIZ_NAME": environ.get(
            "VLLM_NAVI_SCHED_TARGET_BIZ_NAME"
        )
        or biz_name,
        "PYTHONHASHSEED": "0",
    }


def main():
    values = build_environment(
        os.environ,
        sys.argv[1],
        sys.argv[2],
        lambda: socket.gethostbyname(socket.gethostname()),
    )
    # vLLM already supports dictConfig; generate its file-only default before
    # importing vLLM in the API server or any spawned worker process.
    if not os.environ.get("VLLM_LOGGING_CONFIG_PATH"):
        from pathlib import Path
        from .VLLMLogging import write_vllm_logging_config

        path = Path(os.environ.get("NAVI_SCHED_LOG_DIR", "logs")).absolute() / "vllm_logging.json"
        write_vllm_logging_config(path)
        values["VLLM_LOGGING_CONFIG_PATH"] = str(path)
    for name, value in values.items():
        print(f"export {name}={shlex.quote(value)}")


if __name__ == "__main__":
    main()
