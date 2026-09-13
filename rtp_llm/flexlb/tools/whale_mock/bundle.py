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
    overrides = yaml.safe_load(os.environ.get("MOCK_BUNDLE_OVERRIDES_YAML", "{}"))
    numeric_overrides = {
        "prefill",
        "decode",
        "block_size",
        "prefill_block_size",
        "decode_block_size",
        "prefill_kv_pool_blocks",
        "decode_kv_pool_blocks",
        "decode_max_concurrency",
    }
    if not isinstance(overrides, dict) or set(overrides) - numeric_overrides - {
        "mock_heap",
        "master_heap",
    }:
        raise ValueError("unsupported bundle override")
    for role in numeric_overrides:
        if role in overrides and (
            type(overrides[role]) is not int or overrides[role] <= 0
        ):
            raise ValueError(f"{role} must be a positive integer")
    cfg.update(overrides)
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
    performance_path = (cfg_path.parent / cfg["performance"]).resolve()
    performance_json = os.environ.get("MOCK_PERFORMANCE_CONFIG_JSON")
    eos_json = os.environ.get("MOCK_EOS_CONFIG_JSON")
    performance = json.loads(
        performance_json
        if performance_json is not None
        else performance_path.read_text()
    )
    if not isinstance(performance, dict):
        raise ValueError("MOCK_PERFORMANCE_CONFIG_JSON must be a JSON object")
    if performance.get("block_size", cfg["block_size"]) != cfg["block_size"]:
        raise ValueError("performance block_size must match bundle block_size")
    performance["block_size"] = cfg["block_size"]
    if eos_json is not None:
        eos = json.loads(eos_json)
        if not isinstance(eos, dict):
            raise ValueError("MOCK_EOS_CONFIG_JSON must be a JSON object")
        performance.setdefault("decode", {})["eos"] = eos
    if performance_json is not None or eos_json is not None:
        performance_path = runtime / "performance.json"
        performance_path.write_text(json.dumps(performance))
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
                "--whale",
                "true",
                "--whale-bundle",
                "true",
                "--kmonitor",
                str(cfg["kmonitor"]).lower(),
                "--event-loop-threads",
                str(cfg["event_loop_threads"]),
                "--completion-threads",
                str(cfg["completion_threads"]),
                "--n-prefill",
                str(cfg["prefill"]),
                "--n-decode",
                str(cfg["decode"]),
                *(
                    ["--decode-max-concurrency", str(cfg["decode_max_concurrency"])]
                    if "decode_max_concurrency" in cfg
                    else []
                ),
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
                str(performance_path),
                "--master-config",
                str(runtime / "master-config.json"),
                "--block-size",
                str(cfg["block_size"]),
                *(
                    ["--prefill-block-size", str(cfg["prefill_block_size"])]
                    if "prefill_block_size" in cfg
                    else []
                ),
                *(
                    ["--decode-block-size", str(cfg["decode_block_size"])]
                    if "decode_block_size" in cfg
                    else []
                ),
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
