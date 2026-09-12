"""Supervise one master and a local mock cluster as separate JVMs in a test Pod."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "online_eval"))
from flexlb_cfg import render_env, render_process_config


def run():
    if os.environ.get("RTP_LLM_MOCK_BUNDLE") != "1":
        raise ValueError("test-only bundle requires RTP_LLM_MOCK_BUNDLE=1")
    cfg_path = Path(os.environ.get("MOCK_BUNDLE_CONFIG_PATH", ROOT / "bundle.yaml"))
    cfg = yaml.safe_load(cfg_path.read_text())
    if os.environ.get("FETCH_OUTPUT_STREAM", "0") != "0":
        raise ValueError("bundle mode requires FETCH_OUTPUT_STREAM=0")
    if cfg["profile"] not in {"single-batch", "batch-window"}:
        raise ValueError("no-fetch requires BATCH dispatcher")
    http_port = int(os.environ.get("START_PORT", "7001"))
    mock_port = http_port + cfg["mock_port_offset"]
    runtime = Path(os.environ.get("MOCK_BUNDLE_RUN_DIR", "/home/admin/ai-whale/mock"))
    runtime.mkdir(parents=True, exist_ok=False)
    jars = Path(os.environ.get("MOCK_BUNDLE_JAR_DIR", ROOT / "jars"))
    raw = os.environ.get("FLEXLB_CONFIG") or render_env(cfg["profile"])
    (runtime / "master-config.json").write_text(
        render_process_config(
            cfg["profile"], jvm_heap=cfg["master_heap"], raw_config=raw
        )
    )
    children, logs = [], []
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def start(name, command, env):
        log = (runtime / (name + ".log")).open("w")
        logs.append(log)
        p = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        children.append(p)
        return p

    def ready(port, process, host="127.0.0.1"):
        deadline = time.monotonic() + cfg["startup_timeout_s"]
        while time.monotonic() < deadline and not stopping:
            if process.poll() is not None:
                raise RuntimeError("JVM exited before ready")
            try:
                with socket.create_connection((host, port), timeout=1):
                    return
            except OSError:
                time.sleep(0.2)
        raise TimeoutError("startup deadline exceeded")

    try:
        env = os.environ.copy()
        # The control API uses the real Pod address; engine RPCs use unique
        # loopbacks inside this Pod to preserve master engineIp metric identity.
        pod_ip = os.environ.get("POD_IP") or socket.gethostbyname(socket.gethostname())
        if pod_ip.startswith("127.") or pod_ip == "0.0.0.0":
            raise ValueError("bundle requires an advertised Pod IP")
        # The JVM owns engine completion, never waits for frontend output fetching.
        mock = start(
            "mock",
            [
                "java",
                "-Xmx" + cfg["mock_heap"],
                "-jar",
                str(jars / "mock.jar"),
                "--whale", "true",
                "--whale-bundle", "true",
                "--kmonitor", str(cfg["kmonitor"]).lower(),
                "--event-loop-threads", str(cfg["event_loop_threads"]),
                "--completion-threads", str(cfg["completion_threads"]),
                "--n-prefill",
                str(cfg["prefill"]),
                "--n-decode",
                str(cfg["decode"]),
                "--base-grpc-port",
                str(mock_port),
                "--host",
                pod_ip,
                "--auto-fetch",
                "true",
                "--endpoint-file",
                str(runtime / "endpoints.json"),
                "--discovery-file",
                str(runtime / "discovery.json"),
                "--performance",
                str((cfg_path.parent / cfg["performance"]).resolve()),
                "--master-config",
                str(runtime / "master-config.json"),
                "--block-size",
                str(cfg["block_size"]),
                "--prefill-kv-pool-blocks",
                str(cfg["prefill_kv_pool_blocks"]),
                "--decode-kv-pool-blocks",
                str(cfg["decode_kv_pool_blocks"]),
                "--events-file",
                str(runtime / "engine-events.jsonl"),
            ],
            env,
        )
        ready(mock_port - 1, mock, pod_ip)
        env.update(json.loads((runtime / "endpoints.json").read_text())["env"])
        env.update(
            FLEXLB_CONFIG=raw,
            SERVER_PORT=str(http_port),
            MANAGEMENT_SERVER_PORT=str(http_port + 1),
        )
        master = start(
            "master",
            [
                "java",
                "-Xmx" + cfg["master_heap"],
                "-Dreactor.schedulers.defaultBoundedElasticSize=64",
                "-jar",
                str(jars / "master.jar"),
                "--server.port=" + str(http_port),
                "--management.server.port=" + str(http_port + 1),
                "--flexlb.log.path=" + str(runtime / "master-logs"),
            ],
            env,
        )
        ready(http_port + 2, master)
        # This bundle replaces appctl, so it also owns the normal local online hook.
        deadline = time.monotonic() + cfg["startup_timeout_s"]
        while not stopping:
            if master.poll() is not None:
                raise RuntimeError("master exited before application readiness")
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/hook/process_ok", timeout=2
                ) as response:
                    if response.status == 200:
                        break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("master application readiness deadline exceeded")
                time.sleep(0.2)
        if stopping:
            return
        with urllib.request.urlopen(
            f"http://127.0.0.1:{http_port}/hook/after_start",
            timeout=cfg["startup_timeout_s"],
        ) as response:
            response.read()
        with urllib.request.urlopen(
            f"http://127.0.0.1:{http_port}/health", timeout=2
        ) as response:
            response.read()
        (runtime / "identity.json").write_text(
            json.dumps(
                {
                    "mode": "WHALE_BUNDLE_SCHEDULE_ONLY",
                    "fetch_output_stream": False,
                    "master_pid": master.pid,
                    "mock_pid": mock.pid,
                    "master_http_port": http_port,
                    "mock_http_port": mock_port - 1,
                    "configuration": cfg,
                    "source": os.environ.get("BUILD_SOURCE_SHA", "unknown"),
                },
                indent=2,
            )
        )
        while not stopping:
            if any(p.poll() is not None for p in children):
                raise RuntimeError("bundle child exited; terminate its peer")
            time.sleep(0.2)
    finally:
        for p in reversed(children):
            if p.poll() is None:
                p.terminate()
        deadline = time.monotonic() + cfg["shutdown_timeout_s"]
        for p in reversed(children):
            try:
                p.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    run()
