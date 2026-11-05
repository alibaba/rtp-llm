"""Supervise a dedicated Spectrum NaviSched Master deployment."""

import json
import logging
import os
import re
import signal
import subprocess
import sys
import time

LOG = logging.getLogger(__name__)
MAX_CONSECUTIVE_FAILURES = 5
STABLE_RUN_SECONDS = 300


def checked_config(env):
    """Reject a misplaced launcher or a static Prefill node list."""
    identity = {
        "SPECTRUM_WORKSPACE_ID": "NAVI_SCHED_EXPECTED_WORKSPACE_ID",
        "SPECTRUM_APPLICATION_NAME": "NAVI_SCHED_EXPECTED_APPLICATION_NAME",
        "SPECTRUM_DEPLOYMENT_NAME": "NAVI_SCHED_EXPECTED_MASTER_DEPLOYMENT_NAME",
    }
    for actual_name, expected_name in identity.items():
        actual_value = env.get(actual_name, "")
        if not actual_value:
            raise ValueError(f"missing {actual_name} for dedicated Master")
        expected_value = env.get(expected_name, "")
        if expected_value and actual_value != expected_value:
            raise ValueError(f"unexpected {actual_name} for dedicated Master")
    if not env["SPECTRUM_DEPLOYMENT_NAME"].startswith(
        env["SPECTRUM_APPLICATION_NAME"] + "-"
    ):
        raise ValueError("unexpected SPECTRUM_DEPLOYMENT_NAME for application")
    role = env.get("DASHGEN_DEPLOYMENT_ROLE", "")
    if not role:
        raise ValueError("missing DASHGEN_DEPLOYMENT_ROLE")
    expected_role = env.get("NAVI_SCHED_EXPECTED_MASTER_ROLE", "")
    if expected_role and role != expected_role:
        raise ValueError("unexpected DASHGEN_DEPLOYMENT_ROLE for dedicated Master")
    expected = {
        "DEPLOYMENT_NAME": env["SPECTRUM_DEPLOYMENT_NAME"],
        "NAVI_SCHED_KMON_PLATFORM_DISCOVERY": "1",
        "NAVI_SCHED_MASTER_ENABLE_KMON": "1",
    }
    for name, value in expected.items():
        if env.get(name) != value:
            raise ValueError(f"unexpected {name} for dedicated Master")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", env.get("NAVI_SCHED_MASTER_BIZ_NAME", "")):
        raise ValueError("invalid NAVI_SCHED_MASTER_BIZ_NAME")
    ports = {}
    for name in ("NAVI_SCHED_MASTER_GRPC_PORT", "NAVI_SCHED_MASTER_HTTP_PORT"):
        value = env.get(name, "")
        if not value.isdecimal() or not 1 <= int(value) <= 65535:
            raise ValueError(f"invalid {name}")
        ports[name] = int(value)
    if ports["NAVI_SCHED_MASTER_GRPC_PORT"] == ports["NAVI_SCHED_MASTER_HTTP_PORT"]:
        raise ValueError("Master gRPC and HTTP ports must differ")
    agent_port_diff = env.get("NAVI_SCHED_AGENT_GRPC_PORT_DIFF", "")
    if not agent_port_diff.isdecimal() or not 1 <= int(agent_port_diff) <= 65534:
        raise ValueError("invalid NAVI_SCHED_AGENT_GRPC_PORT_DIFF")
    master_service = env.get("NAVI_SCHED_MASTER_SERVICE_ID", "")
    prefill_service = env.get("NAVI_SCHED_PREFILL_SERVICE_ID", "")
    if not re.fullmatch(r"v-[a-z0-9]+", master_service):
        raise ValueError("invalid NAVI_SCHED_MASTER_SERVICE_ID")
    if not re.fullmatch(r"v-[a-z0-9]+", prefill_service):
        raise ValueError("invalid NAVI_SCHED_PREFILL_SERVICE_ID")
    with open(env["NAVI_SCHED_MASTER_CONFIG_FILE"], encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("Master config must be an object")
    runtime = config.get("runtime_config")
    if not isinstance(runtime, dict) or runtime.get("local_nodes") != []:
        raise ValueError("dedicated Master must not use static Prefill nodes")
    expected_subscription = {
        "service_id": prefill_service,
        "grpc_port_diff": int(agent_port_diff),
    }
    if runtime.get("spectrum_subscription") != expected_subscription:
        raise ValueError("dedicated Master must subscribe to target Prefill")
    return config


def _stop_child(child):
    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        LOG.error("Master did not stop after SIGTERM; sending SIGKILL")
        child.kill()
        child.wait()


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    checked_config(os.environ)
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    child_env = dict(os.environ)
    child_env.pop("LD_PRELOAD", None)
    failures = 0
    while not stopping:
        started = time.monotonic()
        child = subprocess.Popen(
            [sys.executable, "-m", "navi_sched.master_process"],
            env=child_env,
        )
        LOG.info("dedicated Master child started pid=%d", child.pid)
        try:
            while not stopping and child.poll() is None:
                try:
                    child.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if stopping:
                _stop_child(child)
        if stopping:
            break
        if time.monotonic() - started >= STABLE_RUN_SECONDS:
            failures = 0
        failures += 1
        LOG.error(
            "dedicated Master exited with status %s (failure %d/%d)",
            child.returncode,
            failures,
            MAX_CONSECUTIVE_FAILURES,
        )
        if failures >= MAX_CONSECUTIVE_FAILURES:
            raise RuntimeError("dedicated Master repeatedly exited")
        deadline = time.monotonic() + min(2 ** (failures - 1), 16)
        while not stopping and time.monotonic() < deadline:
            time.sleep(max(0, min(0.2, deadline - time.monotonic())))


if __name__ == "__main__":
    main()
