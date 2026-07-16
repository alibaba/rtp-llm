#!/usr/bin/env python3
"""Orchestrate inflight-vs-QPS experiment: start system, sweep speeds, save results."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FLEXLB_DIR = SCRIPT_DIR.parent.parent
REPO_ROOT = FLEXLB_DIR.parent.parent

TRACE_FILE = str(SCRIPT_DIR / "data/online_logs/trace_30min.jsonl")
PERFORMANCE_FILE = str(
    SCRIPT_DIR / "data/performance/dsv4_flash_performance.sample.json"
)
PROCESS_CONFIG_FILE = str(SCRIPT_DIR / "data/config/master_fixed_window_220ms_4g.json")
FLEXLB_JAR = str(FLEXLB_DIR / "flexlb-api/target/flexlb-api-1.0.0-SNAPSHOT.jar")

EXPERIMENT_DIR = SCRIPT_DIR / "run" / "inflight_experiment"

N_PREFILL = 2
N_DECODE = 4
MOCK_BASE_GRPC_PORT = 55151
MOCK_HTTP_PORT = MOCK_BASE_GRPC_PORT - 1  # 55150
PREFILL_CACHE_BLOCKS = 6000
DECODE_CACHE_BLOCKS = 3000

FLEXLB_HTTP_PORT = 7001
FLEXLB_MGMT_PORT = 7002

SPEEDS = [3, 5, 7]

DEFAULT_FLEXLB_CONFIG = json.dumps(
    {
        "loadBalanceStrategy": "COST_BASED_PREFILL",
        "decodeLoadBalanceStrategy": "COST_BASED_DECODE",
        "cacheHitMaxCacheKeys": 80000000,
        "cacheHitMetricReportEnabled": True,
        "cacheHitTimeWindowMs": 1800000,
        "cacheHitTraceLogEnabled": False,
        "cacheHitWindowWriteEnabled": True,
        "decodeConcurrencyLimit": 132,
        "flexlbBatchAlgorithm": "fixed_window",
        "flexlbBatchFixedWaitMs": 220,
        "flexlbBatchPredictThresholdMs": 550,
        "flexlbBatchSizeMax": 32,
        "hysteresisBiasPercent": 30,
        "maxQueueSize": 5000,
        "prefillQueueSizeThreshold": 100000,
        "defaultScheduleMode": "BATCH",
        "flexlbBatchFixedMaxInflightBatches": 2,
        "costSloMs": 1000,
        "flexlbBatchMinSize": 8,
        "prefillLbTimeoutMs": 5000,
    }
)
DEFAULT_STRATEGY_CONFIGS = json.dumps(
    {
        "shortestTtft": {"candidatePool": {"mode": "FIXED", "size": 2}},
    }
)

JAVA_MODULE_OPTS = [
    "--add-modules",
    "ALL-SYSTEM",
    "--add-opens",
    "java.base/java.lang=ALL-UNNAMED",
    "--add-opens",
    "java.base/java.lang.invoke=ALL-UNNAMED",
    "--add-opens",
    "java.base/java.util=ALL-UNNAMED",
    "--add-opens",
    "java.base/java.util.concurrent=ALL-UNNAMED",
    "--add-opens=java.base/jdk.internal.misc=ALL-UNNAMED",
    "--add-opens",
    "java.base/java.nio=ALL-UNNAMED",
    "--add-opens",
    "java.base/sun.nio.ch=ALL-UNNAMED",
    "--add-opens",
    "java.instrument/sun.instrument=ALL-UNNAMED",
]


def wait_for_port(host: str, port: int, timeout_s: float = 30.0) -> bool:
    import socket

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def wait_for_http(url: str, timeout_s: float = 30.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def get_inflight() -> dict | None:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{FLEXLB_HTTP_PORT}/rtp_llm/inflight_status", timeout=3.0
        ) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def get_total_inflight() -> int:
    data = get_inflight()
    if not data:
        return -1
    total = data.get("scheduler_inflight", 0)
    return total


def wait_inflight_drain(timeout_s: float = 120.0) -> bool:
    """Wait for scheduler inflight to return to 0."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        inflight = get_total_inflight()
        if inflight == 0:
            print(f"  inflight drained to 0")
            return True
        print(f"  waiting for inflight drain: {inflight}", end="\r")
        time.sleep(2.0)
    print(f"  inflight drain timeout, last value: {get_total_inflight()}")
    return False


def start_mock_engine(experiment_dir: Path) -> subprocess.Popen:
    endpoint_file = str(experiment_dir / "endpoints.json")
    env_file = str(experiment_dir / "flexlb_env.txt")
    log_file = open(str(experiment_dir / "mock_engine.log"), "w")

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "mock_engine_cluster.py"),
        "--n-prefill",
        str(N_PREFILL),
        "--n-decode",
        str(N_DECODE),
        "--base-grpc-port",
        str(MOCK_BASE_GRPC_PORT),
        "--performance",
        PERFORMANCE_FILE,
        "--prefill-cache-blocks",
        str(PREFILL_CACHE_BLOCKS),
        "--decode-cache-blocks",
        str(DECODE_CACHE_BLOCKS),
        "--endpoint-file",
        endpoint_file,
        "--env-file",
        env_file,
    ]

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(SCRIPT_DIR)

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
    )
    print(
        f"mock engine cluster PID={proc.pid}, waiting for port {MOCK_BASE_GRPC_PORT}..."
    )
    if not wait_for_port("127.0.0.1", MOCK_BASE_GRPC_PORT, 20):
        proc.kill()
        raise RuntimeError("mock engine failed to start")
    if not wait_for_http(f"http://127.0.0.1:{MOCK_HTTP_PORT}/snapshot", 15):
        proc.kill()
        raise RuntimeError("mock engine HTTP API not ready")
    print("mock engine cluster started")
    return proc


def read_endpoint_env(endpoint_file: str) -> dict:
    data = json.loads(Path(endpoint_file).read_text(encoding="utf-8"))
    return data.get("env", {})


def read_process_env(config_file: str) -> dict:
    data = json.loads(Path(config_file).read_text(encoding="utf-8"))
    envs = data.get("zone_process_setting", {}).get("process_info", {}).get("envs", [])
    result = {}
    for item in envs:
        if isinstance(item, list) and len(item) == 2:
            result[str(item[0])] = str(item[1])
    return result


def start_flexlb_master(experiment_dir: Path) -> subprocess.Popen:
    endpoint_file = str(experiment_dir / "endpoints.json")
    endpoint_env = read_endpoint_env(endpoint_file)
    process_env = read_process_env(PROCESS_CONFIG_FILE)

    full_env = os.environ.copy()
    full_env.update(endpoint_env)
    full_env.update(process_env)  # process config overrides endpoint env
    full_env["FLEXLB_CONFIG"] = DEFAULT_FLEXLB_CONFIG
    full_env["STRATEGY_CONFIGS"] = DEFAULT_STRATEGY_CONFIGS
    full_env["OTEL_TRACE_SKIP_PATTERN"] = ".*"
    full_env["OTEL_EXPORTER_OTLP_ENDPOINT"] = "none"
    full_env["HIPPO_ROLE"] = "flexlb_eval_master"

    heap_size = process_env.get("FLEXLB_JVM_HEAP_SIZE", "4g")
    log_file = open(str(experiment_dir / "flexlb.log"), "w")

    cmd = [
        "java",
        f"-Xms{heap_size}",
        f"-Xmx{heap_size}",
        *JAVA_MODULE_OPTS,
        "-jar",
        FLEXLB_JAR,
        f"--server.port={FLEXLB_HTTP_PORT}",
        f"--management.server.port={FLEXLB_MGMT_PORT}",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=full_env,
        preexec_fn=os.setsid,
    )
    print(f"flexlb master PID={proc.pid}, waiting for port {FLEXLB_HTTP_PORT}...")
    if not wait_for_port("127.0.0.1", FLEXLB_HTTP_PORT, 60):
        proc.kill()
        raise RuntimeError("flexlb master failed to start")
    if not wait_for_http(
        f"http://127.0.0.1:{FLEXLB_HTTP_PORT}/rtp_llm/inflight_status", 30
    ):
        proc.kill()
        raise RuntimeError("flexlb master HTTP API not ready")
    print("flexlb master started")
    return proc


def run_sweep(speed: int, experiment_dir: Path) -> dict:
    sweep_dir = experiment_dir / f"sweep_{speed}x"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    monitor_output = str(sweep_dir / "monitor.jsonl")
    load_client_dir = str(sweep_dir / "load_client")

    # Start monitor
    monitor_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "stability_monitor.py"),
        "--flexlb-http-addr",
        f"127.0.0.1:{FLEXLB_HTTP_PORT}",
        "--management-port",
        str(FLEXLB_MGMT_PORT),
        "--mock-http-port",
        str(MOCK_HTTP_PORT),
        "--interval",
        "1",
        "--output",
        monitor_output,
    ]
    monitor_env = os.environ.copy()
    monitor_env["PYTHONPATH"] = str(SCRIPT_DIR)
    monitor_log = open(str(sweep_dir / "monitor.log"), "w")
    monitor_proc = subprocess.Popen(
        monitor_cmd,
        stdout=monitor_log,
        stderr=subprocess.STDOUT,
        env=monitor_env,
        preexec_fn=os.setsid,
    )
    print(f"  monitor started PID={monitor_proc.pid}")

    # Run load client
    client_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "flexlb_load_client.py"),
        TRACE_FILE,
        "--flexlb-http-addr",
        f"127.0.0.1:{FLEXLB_HTTP_PORT}",
        "--schedule-mode",
        "batch",
        "--replay-speed",
        str(speed),
        "--max-concurrency",
        os.environ.get("MAX_CONCURRENCY", "16384"),
        "--timeout-ms",
        "3600000",
        "--zero-output-policy",
        "one",
        "--output-dir",
        load_client_dir,
    ]
    client_env = os.environ.copy()
    client_env["PYTHONDONTWRITEBYTECODE"] = "1"
    client_env["PYTHONPATH"] = str(SCRIPT_DIR)
    print(f"  running load client at {speed}x...")
    client_log = open(str(sweep_dir / "client.stdout"), "w")
    client_proc = subprocess.run(
        client_cmd,
        stdout=client_log,
        stderr=subprocess.STDOUT,
        env=client_env,
    )
    print(f"  load client exited with code {client_proc.returncode}")

    # Stop monitor
    time.sleep(3)  # let monitor capture final state
    try:
        os.killpg(os.getpgid(monitor_proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    monitor_proc.wait(timeout=10)
    monitor_log.close()
    print(f"  monitor stopped")

    # Load summary
    summary_path = Path(load_client_dir) / "summary.json"
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

    return {
        "speed": speed,
        "sweep_dir": str(sweep_dir),
        "monitor_file": monitor_output,
        "summary": summary,
    }


def analyze_stability(monitor_file: str) -> dict:
    """Quick stability check from monitor data."""
    records = []
    with open(monitor_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return {"stable": False, "reason": "no monitor data"}

    inflights = [r.get("scheduler_inflight", 0) or 0 for r in records]
    max_inflight = max(inflights)
    final_inflight = inflights[-1] if inflights else 0
    avg_inflight = sum(inflights) / len(inflights) if inflights else 0

    # Check if inflight was still high at the end (not drained)
    last_10pct = inflights[-max(1, len(inflights) // 10) :]
    last_avg = sum(last_10pct) / len(last_10pct) if last_10pct else 0

    # If final inflight is much higher than average, it's growing
    stable = last_avg < avg_inflight * 2 and final_inflight < max_inflight * 0.5

    return {
        "stable": stable,
        "max_inflight": max_inflight,
        "final_inflight": final_inflight,
        "avg_inflight": round(avg_inflight, 1),
        "last_10pct_avg": round(last_avg, 1),
        "num_samples": len(records),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cleanup-only",
        action="store_true",
        help="kill any existing processes on the ports and exit",
    )
    args = parser.parse_args()

    if args.cleanup_only:
        cleanup_ports()
        return

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== Inflight vs QPS Experiment ===")
    print(f"experiment dir: {EXPERIMENT_DIR}")

    mock_proc = None
    flexlb_proc = None

    try:
        # Phase 1: Start system
        print("\n--- Phase 1: Start system ---")
        mock_proc = start_mock_engine(EXPERIMENT_DIR)
        flexlb_proc = start_flexlb_master(EXPERIMENT_DIR)

        # Verify
        inflight = get_inflight()
        print(f"  inflight_status: {json.dumps(inflight, indent=2)[:500]}")

        # Phase 2: Sweep
        print("\n--- Phase 2: Sweep speeds ---")
        results = []
        for speed in SPEEDS:
            print(f"\n  === Sweep {speed}x ===")
            result = run_sweep(speed, EXPERIMENT_DIR)
            stability = analyze_stability(result["monitor_file"])
            result["stability"] = stability
            results.append(result)

            print(f"  stability: {json.dumps(stability, indent=2)}")

            # Wait for inflight drain between speeds
            if speed != SPEEDS[-1]:
                print(f"  waiting for inflight drain before next speed...")
                wait_inflight_drain(120)

        # Print summary
        print("\n--- Sweep Summary ---")
        for r in results:
            s = r.get("summary", {})
            st = r.get("stability", {})
            print(
                f"  {r['speed']}x: offered_qps={s.get('offered_qps', '?')}, "
                f"completed_qps={s.get('completed_qps', '?')}, "
                f"completed={s.get('completed', '?')}/{s.get('total_requests', '?')}, "
                f"max_inflight={st.get('max_inflight', '?')}, "
                f"stable={st.get('stable', '?')}"
            )

        # Save results metadata
        results_file = EXPERIMENT_DIR / "experiment_results.json"
        results_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nresults saved to {results_file}")

    finally:
        # Phase 4: Cleanup
        print("\n--- Phase 4: Cleanup ---")
        for proc, name in [(flexlb_proc, "flexlb"), (mock_proc, "mock_engine")]:
            if proc and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=10)
                    print(f"  {name} (PID={proc.pid}) stopped")
                except Exception:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        print(f"  {name} (PID={proc.pid}) killed")
                    except Exception:
                        pass

    print("\n=== Experiment complete ===")
    print(f"all outputs in: {EXPERIMENT_DIR}")


def cleanup_ports():
    """Kill processes on experiment ports."""
    import subprocess as sp

    for port in [
        FLEXLB_HTTP_PORT,
        FLEXLB_MGMT_PORT,
        MOCK_HTTP_PORT,
        MOCK_BASE_GRPC_PORT,
        MOCK_BASE_GRPC_PORT + 1,
        MOCK_BASE_GRPC_PORT + 2,
        MOCK_BASE_GRPC_PORT + 3,
        MOCK_BASE_GRPC_PORT + 4,
        MOCK_BASE_GRPC_PORT + 5,
    ]:
        result = sp.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
        pids = result.stdout.strip().split()
        for pid in pids:
            if pid:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    print(f"  killed PID {pid} on port {port}")
                except Exception:
                    pass


if __name__ == "__main__":
    main()
