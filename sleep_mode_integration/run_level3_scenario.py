#!/usr/bin/env python3
"""Deterministic level-3 sleep/wake scenario for a local RTP-LLM server."""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

PORT = int(os.environ.get("PORT", sys.argv[1] if len(sys.argv) > 1 else 39080))
GPU_IDS = os.environ.get("GPU", sys.argv[2] if len(sys.argv) > 2 else "3")
MODEL = os.environ.get("MODEL", "deepseek-v4-flash")
BASE = f"http://127.0.0.1:{PORT}"
PROMPT = os.environ.get("PROMPT", "Write one short sentence about reliable serving.")
MAX_CHECKPOINTED_GPU_MIB = int(os.environ.get("MAX_CHECKPOINTED_GPU_MIB", "64"))
MIN_GPU_DROP_MIB = int(os.environ.get("MIN_GPU_DROP_MIB", "512"))
GPU_BASELINE_MIB = int(os.environ.get("GPU_BASELINE_MIB", "0"))
VERIFY_INFERENCE = os.environ.get("VERIFY_INFERENCE", "1").lower() not in {
    "0",
    "false",
    "off",
}
EXPECTED_RANKS = int(
    os.environ.get(
        "EXPECTED_RANKS",
        str(len([gpu_id for gpu_id in GPU_IDS.split(",") if gpu_id.strip()])),
    )
)

results: List[Tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


def http(
    method: str, path: str, body: Optional[Dict[str, Any]] = None, timeout: int = 120
) -> Tuple[int, Dict[str, Any]]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(BASE + path, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode()
            return response.status, json.loads(payload) if payload else {}
    except urllib.error.HTTPError as error:
        try:
            payload = json.loads(error.read().decode())
        except Exception:
            payload = {}
        return error.code, payload
    except Exception as error:
        return -1, {"error": str(error)}


def _nvidia_smi(query: str) -> List[str]:
    result = subprocess.run(
        ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def gpu_memory() -> Tuple[int, Dict[str, int]]:
    wanted = {gpu_id.strip() for gpu_id in GPU_IDS.split(",") if gpu_id.strip()}
    detail: Dict[str, int] = {}
    for line in _nvidia_smi("gpu=index,memory.used"):
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] in wanted:
            detail[fields[0]] = int(fields[1])
    return sum(detail.values()), detail


def gpu_process_memory() -> Dict[int, int]:
    wanted = {gpu_id.strip() for gpu_id in GPU_IDS.split(",") if gpu_id.strip()}
    wanted_uuids = set()
    for line in _nvidia_smi("gpu=index,uuid"):
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] in wanted:
            wanted_uuids.add(fields[1])

    memory: Dict[int, int] = {}
    for line in _nvidia_smi("compute-apps=gpu_uuid,pid,used_gpu_memory"):
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3 and fields[0] in wanted_uuids:
            pid = int(fields[1])
            memory[pid] = memory.get(pid, 0) + int(fields[2])
    return memory


def process_gpu_memory(pid: int) -> int:
    return gpu_process_memory().get(pid, 0)


def infer(max_tokens: int = 24, timeout: int = 180) -> Tuple[int, Dict[str, Any]]:
    return http(
        "POST",
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_k": 1,
        },
        timeout=timeout,
    )


def response_text(response: Dict[str, Any]) -> str:
    return response.get("choices", [{}])[0].get("message", {}).get("content", "")


def main() -> None:
    code, _ = http("GET", "/health")
    record("health", code == 200, f"code={code}")
    if code != 200:
        raise SystemExit(1)

    baseline = ""
    if VERIFY_INFERENCE:
        code, response = infer()
        baseline = response_text(response) if code == 200 else ""
        record(
            "baseline inference",
            code == 200 and bool(response.get("choices")),
            f"code={code} text={baseline[:80]!r}",
        )
        if code != 200 or not response.get("choices"):
            raise SystemExit(1)

        warm_code, warm_response = infer()
        record(
            "warm baseline inference",
            warm_code == 200 and response_text(warm_response) == baseline,
            f"code={warm_code} match={response_text(warm_response) == baseline}",
        )
        if warm_code != 200:
            raise SystemExit(1)

    code, status = http("GET", "/sleep_status")
    backend_pid = int(status.get("process_id", 0))
    initial_epoch = int(status.get("sleep_epoch", 0))
    record(
        "initial sleep_status RUNNING",
        code == 200
        and status.get("state") == "RUNNING"
        and 3 in status.get("supported_levels", [])
        and backend_pid > 0,
        f"code={code} pid={backend_pid} status={status}",
    )

    running_total, running_detail = gpu_memory()
    running_processes = gpu_process_memory()
    running_process = running_processes.get(backend_pid, 0)
    print(
        f"    GPU RUNNING: total={running_total} MiB detail={running_detail} "
        f"backend_pid={backend_pid} process={running_process} MiB "
        f"rank_processes={running_processes}",
        flush=True,
    )
    record(
        "all rank GPU processes discovered",
        len(running_processes) == EXPECTED_RANKS and backend_pid in running_processes,
        f"expected={EXPECTED_RANKS} processes={running_processes}",
    )
    if len(running_processes) != EXPECTED_RANKS:
        raise SystemExit(1)

    started = time.time()
    code, response = http(
        "POST",
        "/sleep",
        {
            "level": 3,
            "mode": "wait",
            "timeout_ms": 30000,
            "reason": "level3-integration-test",
        },
        timeout=300,
    )
    sleep_ms = (time.time() - started) * 1000
    record(
        "level-3 sleep rpc",
        code == 200 and response.get("status") == "ok",
        f"code={code} response={response} took={sleep_ms:.0f}ms",
    )

    code, status = http("GET", "/sleep_status")
    checkpoint_pids = [int(pid) for pid in status.get("process_ids", [])]
    record(
        "status CHECKPOINTED",
        code == 200
        and status.get("state") == "CHECKPOINTED"
        and status.get("gpu_resource_state") == "RELEASED"
        and status.get("device_kv_cache_valid") is False
        and set(checkpoint_pids) == set(running_processes),
        f"code={code} status={status}",
    )

    time.sleep(2)
    checkpointed_total, checkpointed_detail = gpu_memory()
    checkpointed_processes = gpu_process_memory()
    checkpointed_process = checkpointed_processes.get(backend_pid, 0)
    dropped = running_total - checkpointed_total
    print(
        f"    GPU CHECKPOINTED: total={checkpointed_total} MiB "
        f"detail={checkpointed_detail} backend_pid={backend_pid} "
        f"process={checkpointed_process} MiB rank_processes={checkpointed_processes} "
        f"drop={dropped} MiB",
        flush=True,
    )
    record(
        "backend PID GPU memory released",
        not checkpointed_processes,
        f"checkpoint_pids={checkpoint_pids} gpu_processes={checkpointed_processes}",
    )
    record(
        "GPU memory near zero",
        checkpointed_total <= GPU_BASELINE_MIB + MAX_CHECKPOINTED_GPU_MIB
        and dropped >= MIN_GPU_DROP_MIB,
        f"RUNNING={running_total}MiB CHECKPOINTED={checkpointed_total}MiB "
        f"drop={dropped}MiB baseline={GPU_BASELINE_MIB}MiB "
        f"slack={MAX_CHECKPOINTED_GPU_MIB}MiB",
    )

    if VERIFY_INFERENCE:
        code, response = infer(max_tokens=8, timeout=30)
        record(
            "inference rejected while CHECKPOINTED",
            code != 200,
            f"code={code} response={str(response)[:180]}",
        )

    code, response = http("POST", "/sleep", {"level": 3}, timeout=30)
    record(
        "level-3 sleep idempotent",
        code == 200 and response.get("status") == "ok",
        f"code={code} response={response}",
    )

    started = time.time()
    code, response = http("POST", "/wake_up", {}, timeout=300)
    wake_ms = (time.time() - started) * 1000
    record(
        "wake_up rpc",
        code == 200 and response.get("status") == "ok",
        f"code={code} response={response} took={wake_ms:.0f}ms",
    )

    time.sleep(2)
    resumed_total, resumed_detail = gpu_memory()
    resumed_processes = gpu_process_memory()
    resumed_process = resumed_processes.get(backend_pid, 0)
    print(
        f"    GPU RESUMED: total={resumed_total} MiB detail={resumed_detail} "
        f"backend_pid={backend_pid} process={resumed_process} MiB "
        f"rank_processes={resumed_processes}",
        flush=True,
    )

    resumed_text = ""
    if VERIFY_INFERENCE:
        code, response = infer()
        resumed_text = response_text(response) if code == 200 else ""
        record(
            "post-wake inference",
            code == 200 and bool(response.get("choices")),
            f"code={code} text={resumed_text[:80]!r}",
        )
    warmed_resumed_total, warmed_resumed_detail = gpu_memory()
    warmed_resumed_processes = gpu_process_memory()
    warmed_resumed_process = warmed_resumed_processes.get(backend_pid, 0)
    print(
        f"    GPU RESUMED+INFERENCE: total={warmed_resumed_total} MiB "
        f"detail={warmed_resumed_detail} backend_pid={backend_pid} "
        f"process={warmed_resumed_process} MiB "
        f"rank_processes={warmed_resumed_processes}",
        flush=True,
    )
    memory_restored = (
        set(warmed_resumed_processes) == set(running_processes)
        and warmed_resumed_process > 0
    )
    if VERIFY_INFERENCE:
        memory_restored = (
            memory_restored and warmed_resumed_total >= running_total - 512
        )
    record(
        "GPU memory restored",
        memory_restored,
        f"RUNNING={running_total}MiB RESUMED={warmed_resumed_total}MiB",
    )
    if VERIFY_INFERENCE:
        record(
            "greedy output matches baseline",
            resumed_text == baseline,
            f"match={resumed_text == baseline}",
        )

        code, response = infer()
        second_text = response_text(response) if code == 200 else ""
        record(
            "second post-wake inference matches",
            code == 200 and second_text == baseline,
            f"code={code} match={second_text == baseline}",
        )

    code, _ = http("GET", "/health")
    record("post-wake health", code == 200, f"code={code}")

    code, status = http("GET", "/sleep_status")
    record(
        "final status RUNNING",
        code == 200
        and status.get("state") == "RUNNING"
        and int(status.get("sleep_epoch", 0)) == initial_epoch + 1
        and status.get("kv_memory_state") == "ACTIVE",
        f"code={code} status={status}",
    )

    print("\n===== LEVEL-3 SUMMARY =====")
    for name, ok, _ in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"{passed}/{len(results)} passed")
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
