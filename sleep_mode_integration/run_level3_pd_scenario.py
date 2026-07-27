#!/usr/bin/env python3
"""Level-3 sleep/wake E2E for a local PREFILL/DECODE deployment."""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import grpc
import pynvml

import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 as pb2
import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc as pb2_grpc

PREFILL_PORT = int(os.environ.get("PREFILL_PORT", "22000"))
DECODE_PORT = int(os.environ.get("DECODE_PORT", "21000"))
PREFILL_GPUS = os.environ.get("PREFILL_GPUS", "6,7")
DECODE_GPUS = os.environ.get("DECODE_GPUS", "4,5")
MODEL = os.environ.get("MODEL", "deepseek-v4-flash")
PROMPT = os.environ.get("PROMPT", "What is the capital of France?")
EXPECTED_TEXT = os.environ.get("EXPECTED_TEXT", "The capital of France is Paris.")
SLEEP_WAKE_CYCLES = int(os.environ.get("SLEEP_WAKE_CYCLES", "3"))
# Sleep level under test. Must equal the server's startup SLEEP_MODE_LEVEL — the
# backend rejects a mismatched level (INVALID_ARGUMENT). Level 3 alone
# process-checkpoints (state CHECKPOINTED, physical GPU memory -> ~0); levels 1
# and 2 pause/release GPU-side KV+weights but keep the process resident
# (state SLEEPING), so their physical-memory floor is not asserted.
SLEEP_LEVEL = int(os.environ.get("SLEEP_LEVEL", "3"))
if SLEEP_LEVEL not in (1, 2, 3):
    raise ValueError("SLEEP_LEVEL must be 1, 2, or 3")
IS_CHECKPOINT_LEVEL = SLEEP_LEVEL == 3
EXPECTED_RANKS_PER_ROLE = int(os.environ.get("EXPECTED_RANKS_PER_ROLE", "2"))
GPU_BASELINE_MIB = int(os.environ.get("GPU_BASELINE_MIB", "0"))
MAX_CHECKPOINTED_GPU_MIB = int(os.environ.get("MAX_CHECKPOINTED_GPU_MIB", "64"))
MEMORY_SETTLE_SECONDS = float(os.environ.get("MEMORY_SETTLE_SECONDS", "2"))
# A CUDA process checkpoint releases each GPU's user-space memory, but the
# driver's physical release can lag by a few seconds and is not synchronized
# across GPUs (observed: one rank GPU still at ~629MiB at +2s, self-drained to
# ~3MiB shortly after). Re-sample up to this many times so a benign release lag
# is tolerated while a true leak (memory that never drains) still fails.
CHECKPOINT_MEM_SETTLE_ATTEMPTS = int(
    os.environ.get("CHECKPOINT_MEM_SETTLE_ATTEMPTS", "8")
)
SUMMARY_PATH = os.environ.get("SUMMARY_PATH", "")
# On a post-wake inference failure, freeze the live scene instead of only
# recording code+text: re-query sleep status and retry the inference a few times
# to tell a transient wake/admission race (clears on retry) from a stuck engine
# (persists). Written to DIAG_DIR (defaults next to SUMMARY_PATH, else /tmp).
POSTFAIL_RETRIES = int(os.environ.get("POSTFAIL_RETRIES", "6"))
POSTFAIL_RETRY_GAP = float(os.environ.get("POSTFAIL_RETRY_GAP", "3"))
DIAG_DIR = os.environ.get("DIAG_DIR", "")

if SLEEP_WAKE_CYCLES < 1:
    raise ValueError("SLEEP_WAKE_CYCLES must be at least 1")
if EXPECTED_RANKS_PER_ROLE < 1:
    raise ValueError("EXPECTED_RANKS_PER_ROLE must be at least 1")

ROLE_BASES = {
    "prefill": f"http://127.0.0.1:{PREFILL_PORT}",
    "decode": f"http://127.0.0.1:{DECODE_PORT}",
}
ROLE_GPUS = {"prefill": PREFILL_GPUS, "decode": DECODE_GPUS}
ROLE_CONTROL_ADDRESSES = {
    "prefill": [
        address.strip()
        for address in os.environ.get(
            "PREFILL_CONTROL_ADDRESSES",
            f"127.0.0.1:{PREFILL_PORT + 1},127.0.0.1:{PREFILL_PORT + 10}",
        ).split(",")
        if address.strip()
    ],
    "decode": [
        address.strip()
        for address in os.environ.get(
            "DECODE_CONTROL_ADDRESSES",
            f"127.0.0.1:{DECODE_PORT + 1},127.0.0.1:{DECODE_PORT + 10}",
        ).split(",")
        if address.strip()
    ],
}
SLEEP_ROLE_ORDER = ("decode", "prefill")
WAKE_ROLE_ORDER = ("prefill", "decode")

results: List[Tuple[str, bool, str]] = []
cycle_summaries: List[Dict[str, Any]] = []
initial_physical_gpu_memory_mib: Dict[str, int] = {}


class ScenarioStepError(RuntimeError):
    pass


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


def http(
    role: str,
    method: str,
    path: str,
    body: Optional[Dict[str, Any]] = None,
    timeout: int = 120,
) -> Tuple[int, Dict[str, Any]]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(ROLE_BASES[role] + path, data=data, method=method)
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


def infer(timeout: int = 600) -> Tuple[int, Dict[str, Any]]:
    return http(
        "prefill",
        "POST",
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 10,
            "temperature": 0.0,
            "top_p": 0,
            "top_k": 1,
        },
        timeout=timeout,
    )


def response_text(response: Dict[str, Any]) -> str:
    return response.get("choices", [{}])[0].get("message", {}).get("content", "")


def _gpu_ids(gpu_ids: str) -> Set[str]:
    return {gpu.strip() for gpu in gpu_ids.split(",") if gpu.strip()}


def all_gpu_ids() -> Set[str]:
    return _gpu_ids(PREFILL_GPUS) | _gpu_ids(DECODE_GPUS)


def _device_used_mib(handle) -> int:
    # nvmlDeviceGetMemoryInfo (v1) reports used = total - free, which INCLUDES
    # the NVIDIA driver-reserved region (~900MiB/GPU on this box: page tables,
    # ECC carveout, firmware, NVLS fabric). That floor exists on any idle GPU,
    # so a checkpointed CUDA process would misleadingly read ~930MiB "used".
    # v2's `used` excludes reserved, giving the TRUE user-space device
    # allocation (which drops to ~0 after a CUDA process checkpoint).
    try:
        info = pynvml.nvmlDeviceGetMemoryInfo(handle, version=pynvml.nvmlMemory_v2)
        return int(info.used // (1024 * 1024))
    except (TypeError, AttributeError, pynvml.NVMLError):
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(info.used // (1024 * 1024))


def gpu_memory(gpu_ids: str) -> Dict[str, int]:
    wanted = _gpu_ids(gpu_ids)
    memory: Dict[str, int] = {}
    pynvml.nvmlInit()
    try:
        for gpu in sorted(wanted, key=int):
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu))
            memory[gpu] = _device_used_mib(handle)
    finally:
        pynvml.nvmlShutdown()
    return memory


def gpu_processes_by_gpu() -> Dict[str, Dict[int, int]]:
    processes: Dict[str, Dict[int, int]] = {}
    pynvml.nvmlInit()
    try:
        for gpu in sorted(all_gpu_ids(), key=int):
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu))
            gpu_processes: Dict[int, int] = {}
            for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                used_bytes = int(process.usedGpuMemory or 0)
                gpu_processes[int(process.pid)] = used_bytes // (1024 * 1024)
            processes[gpu] = gpu_processes
    finally:
        pynvml.nvmlShutdown()
    return processes


def gpu_process_memory() -> Dict[int, int]:
    memory: Dict[int, int] = {}
    for processes in gpu_processes_by_gpu().values():
        for pid, used in processes.items():
            memory[pid] = memory.get(pid, 0) + used
    return memory


def memory_snapshot(role_pids: Dict[str, Set[int]]) -> Dict[str, Any]:
    processes_by_gpu = gpu_processes_by_gpu()
    physical_gpu_memory = gpu_memory(",".join(sorted(all_gpu_ids())))
    rank_memory_by_gpu: Dict[str, Dict[str, int]] = {}
    rank_memory_total: Dict[str, int] = {}
    rank_roles: Dict[str, str] = {}

    for role, pids in role_pids.items():
        role_gpus = _gpu_ids(ROLE_GPUS[role])
        for pid in sorted(pids):
            by_gpu = {
                gpu: processes_by_gpu.get(gpu, {}).get(pid, 0)
                for gpu in sorted(role_gpus)
            }
            rank_memory_by_gpu[str(pid)] = by_gpu
            rank_memory_total[str(pid)] = sum(by_gpu.values())
            rank_roles[str(pid)] = role

    return {
        "rank_roles": rank_roles,
        "rank_process_memory_mib": rank_memory_total,
        "rank_process_memory_by_physical_gpu_mib": rank_memory_by_gpu,
        "physical_gpu_memory_mib": physical_gpu_memory,
    }


def status(role: str) -> Tuple[int, Dict[str, Any]]:
    return http(role, "GET", "/sleep_status")


def backend_statuses(role: str) -> List[Dict[str, Any]]:
    statuses: List[Dict[str, Any]] = []
    for address in ROLE_CONTROL_ADDRESSES[role]:
        channel = grpc.insecure_channel(address)
        try:
            response = pb2_grpc.RpcServiceStub(channel).GetSleepStatus(
                pb2.EmptyPB(), timeout=5
            )
        finally:
            channel.close()
        statuses.append(
            {
                "address": address,
                "state": response.state,
                "process_id": int(response.process_id),
                "process_starttime": int(response.process_starttime),
            }
        )
    return statuses


def process_starttime(pid: int) -> int:
    with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as stat_file:
        stat = stat_file.read().strip()
    command_end = stat.rfind(")")
    if command_end < 0:
        raise RuntimeError(f"invalid /proc/{pid}/stat")
    fields_after_command = stat[command_end + 2 :].split()
    if len(fields_after_command) <= 19:
        raise RuntimeError(f"incomplete /proc/{pid}/stat")
    return int(fields_after_command[19])


def identities_preserved(
    statuses: List[Dict[str, Any]], identities: Dict[int, int]
) -> bool:
    return all(
        item["process_id"] in identities
        and item["process_starttime"] == identities[item["process_id"]]
        and process_starttime(item["process_id"]) == identities[item["process_id"]]
        for item in statuses
    )


def require_running_statuses() -> (
    Tuple[Dict[str, Set[int]], Dict[int, int], Dict[str, int], Dict[str, Any]]
):
    pids: Dict[str, Set[int]] = {}
    identities: Dict[int, int] = {}
    initial_epochs: Dict[str, int] = {}
    backend_details: Dict[str, Any] = {}
    for role in ("decode", "prefill"):
        code, payload = status(role)
        raw_statuses = backend_statuses(role)
        current_pids = {item["process_id"] for item in raw_statuses}
        identity_ok = all(
            item["process_id"] > 0
            and item["process_starttime"] == process_starttime(item["process_id"])
            for item in raw_statuses
        )
        ok = (
            code == 200
            and payload.get("state") == "RUNNING"
            and SLEEP_LEVEL in payload.get("supported_levels", [])
            and len(current_pids) == EXPECTED_RANKS_PER_ROLE
            and len(raw_statuses) == EXPECTED_RANKS_PER_ROLE
            and all(item["state"] == "RUNNING" for item in raw_statuses)
            and identity_ok
        )
        record(
            f"{role} initial RUNNING",
            ok,
            f"code={code} rank_pids={sorted(current_pids)} "
            f"reported_pid={payload.get('process_id')} "
            f"state={payload.get('state')}",
        )
        pids[role] = current_pids
        initial_epochs[role] = int(payload.get("sleep_epoch", 0))
        backend_details[role] = raw_statuses
        identities.update(
            {item["process_id"]: item["process_starttime"] for item in raw_statuses}
        )
    expected_total = EXPECTED_RANKS_PER_ROLE * len(pids)
    all_pids = pids["decode"] | pids["prefill"]
    record(
        "all rank processes distinct",
        len(all_pids) == expected_total and not (pids["decode"] & pids["prefill"]),
        f"expected={expected_total} pids={pids}",
    )
    return pids, identities, initial_epochs, backend_details


def sleep_role(role: str, cycle: int) -> Dict[str, Any]:
    started = time.time()
    code, payload = http(
        role,
        "POST",
        "/sleep",
        {
            "level": SLEEP_LEVEL,
            "mode": "wait",
            "timeout_ms": 300000,
            "reason": f"pd-level{SLEEP_LEVEL}-e2e-cycle-{cycle}",
        },
        timeout=600,
    )
    duration_seconds = time.time() - started
    ok = code == 200 and payload.get("status") == "ok"
    record(
        f"cycle {cycle} {role} level-{SLEEP_LEVEL} sleep",
        ok,
        f"code={code} took={duration_seconds:.3f}s response={payload}",
    )
    return {
        "code": code,
        "ok": ok,
        "duration_seconds": round(duration_seconds, 3),
        "response": payload,
    }


def wake_role(role: str, cycle: int) -> Dict[str, Any]:
    started = time.time()
    code, payload = http(role, "POST", "/wake_up", {}, timeout=900)
    duration_seconds = time.time() - started
    ok = code == 200 and payload.get("status") == "ok"
    record(
        f"cycle {cycle} {role} wake",
        ok,
        f"code={code} took={duration_seconds:.3f}s response={payload}",
    )
    return {
        "code": code,
        "ok": ok,
        "duration_seconds": round(duration_seconds, 3),
        "response": payload,
    }


def sleep_roles(
    cycle: int,
    sleeping_roles: Set[str],
    wake_attempted_roles: Set[str],
    wake_blocked_roles: Set[str],
    sleep_results: Dict[str, Any],
) -> None:
    for role in SLEEP_ROLE_ORDER:
        sleep_result = sleep_role(role, cycle)
        sleep_results[role] = sleep_result
        if not sleep_result["ok"]:
            raise ScenarioStepError(f"{role} failed to enter level-3 sleep")
        sleeping_roles.add(role)
        wake_attempted_roles.discard(role)
        wake_blocked_roles.discard(role)


def _block_dependent_wakes(
    failed_role: str, sleeping_roles: Set[str], wake_blocked_roles: Set[str]
) -> None:
    failed_index = WAKE_ROLE_ORDER.index(failed_role)
    wake_blocked_roles.update(
        role for role in WAKE_ROLE_ORDER[failed_index + 1 :] if role in sleeping_roles
    )


def wake_roles(
    cycle: int,
    sleeping_roles: Set[str],
    wake_attempted_roles: Set[str],
    wake_blocked_roles: Set[str],
    wake_results: Dict[str, Any],
    *,
    cleanup: bool = False,
) -> None:
    for role in WAKE_ROLE_ORDER:
        if (
            role not in sleeping_roles
            or role in wake_attempted_roles
            or role in wake_blocked_roles
        ):
            continue

        # A failed restore may already have consumed the checkpoint manifest.
        # Mark the attempt before issuing the request so cleanup never retries it.
        wake_attempted_roles.add(role)
        try:
            wake_result = wake_role(role, cycle)
        except Exception:
            _block_dependent_wakes(role, sleeping_roles, wake_blocked_roles)
            raise
        wake_results[role] = wake_result
        if cleanup:
            record(
                f"cleanup {role} wake",
                wake_result["ok"],
                f"response={wake_result['response']}",
            )
        if not wake_result["ok"]:
            _block_dependent_wakes(role, sleeping_roles, wake_blocked_roles)
            raise ScenarioStepError(f"{role} failed to wake")
        sleeping_roles.discard(role)


def checked_inference(name: str, expected: str, timeout: int = 600) -> Dict[str, Any]:
    code, response = infer(timeout=timeout)
    text = response_text(response) if code == 200 else ""
    ok = code == 200 and text == expected and text == EXPECTED_TEXT
    record(name, ok, f"code={code} text={text!r}")
    # Keep the full body so a failure carries error_code/message/aux_info
    # (a raw 200-shape response is large but harmless for the summary).
    return {"code": code, "text": text, "matches_expected": ok, "response": response}


def capture_postwake_failure(
    cycle: int, baseline: str, first: Dict[str, Any]
) -> Dict[str, Any]:
    """Freeze the scene on a post-wake inference failure.

    Records the first failing body, re-queries the frontend + per-rank backend
    sleep status, then retries the inference so a transient wake/admission race
    (clears on retry) is distinguished from a stuck engine (persists). Persisted
    standalone because the caller raises (which skips the summary write).
    """
    diag: Dict[str, Any] = {
        "cycle": cycle,
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "first_attempt": first,
        "status_after_failure": {},
        "retries": [],
        "cleared_on_retry": None,
    }
    for role in ("decode", "prefill"):
        code, payload = status(role)
        try:
            backends = backend_statuses(role)
        except Exception as exc:  # grpc may transiently fail mid-transition
            backends = [{"error": str(exc)}]
        diag["status_after_failure"][role] = {
            "frontend_code": code,
            "frontend": payload,
            "backends": backends,
        }
    for attempt in range(1, max(0, POSTFAIL_RETRIES) + 1):
        time.sleep(POSTFAIL_RETRY_GAP)
        code, response = infer()
        text = response_text(response) if code == 200 else ""
        matches = code == 200 and text == baseline and text == EXPECTED_TEXT
        diag["retries"].append(
            {
                "attempt": attempt,
                "code": code,
                "text": text,
                "matches_expected": matches,
                "response": response,
            }
        )
        record(
            f"cycle {cycle} post-wake inference retry {attempt}",
            matches,
            f"code={code} text={text!r}",
        )
        if matches:
            diag["cleared_on_retry"] = attempt
            break
    out_dir = DIAG_DIR or (os.path.dirname(SUMMARY_PATH) if SUMMARY_PATH else "/tmp")
    try:
        path = os.path.join(out_dir, f"postwake_failure_cycle{cycle}.json")
        Path(path).write_text(json.dumps(diag, indent=2) + "\n", encoding="utf-8")
        print(f"POSTWAKE_FAILURE_DIAG written -> {path}", flush=True)
    except Exception as exc:
        print(f"failed to write postwake diag: {exc}", flush=True)
    return diag


def check_running_memory(
    cycle: int, phase: str, snapshot: Dict[str, Any], expected_pid_count: int
) -> bool:
    rank_memory = snapshot["rank_process_memory_mib"]
    ok = len(rank_memory) == expected_pid_count and all(
        value > 0 for value in rank_memory.values()
    )
    record(
        f"cycle {cycle} all rank GPU memory present {phase}",
        ok,
        f"memory={rank_memory}",
    )
    return ok


def _checkpoint_memory_released(
    snapshot: Dict[str, Any], expected_pid_count: int
) -> Tuple[bool, bool]:
    """(ranks_released, physical_released) for a checkpoint snapshot, no logging."""
    rank_memory = snapshot["rank_process_memory_mib"]
    physical_memory = snapshot["physical_gpu_memory_mib"]
    ranks_released = len(rank_memory) == expected_pid_count and all(
        value == 0 for value in rank_memory.values()
    )
    # physical_memory is true user-space device usage (driver-reserved
    # excluded), so a successful checkpoint drives every role GPU to ~0. Assert
    # against a small absolute floor rather than the running-time baseline.
    physical_released = (
        len(physical_memory) == len(all_gpu_ids())
        and set(physical_memory) == set(initial_physical_gpu_memory_mib)
        and all(
            value <= GPU_BASELINE_MIB + MAX_CHECKPOINTED_GPU_MIB
            for value in physical_memory.values()
        )
    )
    return ranks_released, physical_released


def settled_checkpoint_snapshot(
    role_pids: Dict[str, Set[int]], expected_pid_count: int
) -> Dict[str, Any]:
    """Re-sample the checkpoint memory until it has settled or attempts run out.

    Tolerates the driver's lagged/unsynchronized physical release after a CUDA
    process checkpoint; returns the first fully-released snapshot, else the last.
    """
    snapshot = memory_snapshot(role_pids)
    for _ in range(max(1, CHECKPOINT_MEM_SETTLE_ATTEMPTS)):
        ranks_released, physical_released = _checkpoint_memory_released(
            snapshot, expected_pid_count
        )
        if ranks_released and physical_released:
            return snapshot
        time.sleep(MEMORY_SETTLE_SECONDS)
        snapshot = memory_snapshot(role_pids)
    return snapshot


def check_checkpoint_memory(
    cycle: int, snapshot: Dict[str, Any], expected_pid_count: int
) -> bool:
    rank_memory = snapshot["rank_process_memory_mib"]
    physical_memory = snapshot["physical_gpu_memory_mib"]
    ranks_released, physical_released = _checkpoint_memory_released(
        snapshot, expected_pid_count
    )
    record(
        f"cycle {cycle} all rank PIDs released GPU memory",
        ranks_released,
        f"memory={rank_memory}",
    )
    record(
        f"cycle {cycle} physical GPU memory released to ~0 at checkpoint",
        physical_released,
        f"user_space_memory={physical_memory} "
        f"floor={GPU_BASELINE_MIB + MAX_CHECKPOINTED_GPU_MIB}MiB (driver-reserved excluded)",
    )
    return ranks_released and physical_released


def run_cycle(
    cycle: int,
    baseline: str,
    pids: Dict[str, Set[int]],
    identities: Dict[int, int],
    initial_epochs: Dict[str, int],
    sleeping_roles: Set[str],
    wake_attempted_roles: Set[str],
    wake_blocked_roles: Set[str],
) -> None:
    expected_pid_count = sum(len(role_pids) for role_pids in pids.values())
    cycle_summary: Dict[str, Any] = {"cycle": cycle}
    cycle_summaries.append(cycle_summary)

    cycle_summary["pre_sleep_inference"] = checked_inference(
        f"cycle {cycle} pre-sleep inference matches baseline and golden", baseline
    )
    if not cycle_summary["pre_sleep_inference"]["matches_expected"]:
        raise ScenarioStepError("pre-sleep inference validation failed")
    running_snapshot = memory_snapshot(pids)
    cycle_summary["before_sleep_memory"] = running_snapshot
    if not check_running_memory(
        cycle, "before sleep", running_snapshot, expected_pid_count
    ):
        raise ScenarioStepError("pre-sleep GPU memory validation failed")

    cycle_summary["sleep"] = {}
    sleep_roles(
        cycle,
        sleeping_roles,
        wake_attempted_roles,
        wake_blocked_roles,
        cycle_summary["sleep"],
    )

    cycle_summary["checkpointed_status"] = {}
    sleep_status_ok = True
    # Level 3 process-checkpoints (frontend-synthesized CHECKPOINTED, with a
    # process_ids manifest); levels 1 and 2 pause GPU resources but keep the
    # process resident (backend SLEEPING, per-rank process_id via gRPC).
    expected_sleep_state = "CHECKPOINTED" if IS_CHECKPOINT_LEVEL else "SLEEPING"
    for role in ("decode", "prefill"):
        code, payload = status(role)
        cycle_summary["checkpointed_status"][role] = payload
        role_slept = (
            code == 200
            and payload.get("state") == expected_sleep_state
            and payload.get("gpu_resource_state") == "RELEASED"
            and payload.get("device_kv_cache_valid") is False
            and SLEEP_LEVEL in payload.get("supported_levels", [])
        )
        record(
            f"cycle {cycle} {role} {expected_sleep_state}",
            role_slept,
            f"code={code} state={payload.get('state')} "
            f"gpu_resource_state={payload.get('gpu_resource_state')} "
            f"kv_memory_state={payload.get('kv_memory_state')} "
            f"supported_levels={payload.get('supported_levels')}",
        )
        # Rank identity: L3 exposes the checkpoint pid manifest; L1/L2 keep the
        # live ranks reachable over gRPC, so read their pids from the backends.
        if IS_CHECKPOINT_LEVEL:
            observed_pids = {int(pid) for pid in payload.get("process_ids", [])}
        else:
            observed_pids = {item["process_id"] for item in backend_statuses(role)}
        identities_ok = observed_pids == pids[role] and all(
            process_starttime(pid) == identities[pid] for pid in observed_pids
        )
        record(
            f"cycle {cycle} {role} sleeping identities preserved",
            identities_ok,
            f"expected={sorted(pids[role])} observed={sorted(observed_pids)}",
        )
        sleep_status_ok = sleep_status_ok and role_slept and identities_ok

    if not sleep_status_ok:
        raise ScenarioStepError("sleep status validation failed")

    time.sleep(MEMORY_SETTLE_SECONDS)
    if IS_CHECKPOINT_LEVEL:
        # Poll through the driver's lagged physical release before asserting.
        checkpoint_snapshot = settled_checkpoint_snapshot(pids, expected_pid_count)
        cycle_summary["checkpointed_memory"] = checkpoint_snapshot
        if not check_checkpoint_memory(cycle, checkpoint_snapshot, expected_pid_count):
            raise ScenarioStepError("checkpoint GPU memory validation failed")
    else:
        # L1/L2 do not free device memory to ~0 (process stays resident); record
        # the sleeping footprint for inspection without asserting a floor.
        checkpoint_snapshot = memory_snapshot(pids)
        cycle_summary["checkpointed_memory"] = checkpoint_snapshot
        record(
            f"cycle {cycle} sleeping GPU footprint recorded (level {SLEEP_LEVEL})",
            True,
            f"rank_memory={checkpoint_snapshot['rank_process_memory_mib']} "
            f"physical={checkpoint_snapshot['physical_gpu_memory_mib']}",
        )

    code, payload = infer(timeout=30)
    rejected = code != 200
    cycle_summary["sleeping_inference"] = {
        "code": code,
        "rejected": rejected,
        "response": payload,
    }
    record(
        f"cycle {cycle} inference rejected while both roles sleep",
        rejected,
        f"code={code} response={str(payload)[:200]}",
    )
    if not rejected:
        raise ScenarioStepError("inference was accepted while roles were sleeping")

    cycle_summary["wake"] = {}
    wake_roles(
        cycle,
        sleeping_roles,
        wake_attempted_roles,
        wake_blocked_roles,
        cycle_summary["wake"],
    )

    cycle_summary["running_status"] = {}
    running_status_ok = True
    for role in ("decode", "prefill"):
        code, payload = status(role)
        raw_statuses = backend_statuses(role)
        resumed_pids = {item["process_id"] for item in raw_statuses}
        cycle_summary["running_status"][role] = {
            "frontend": payload,
            "backends": raw_statuses,
        }
        role_running = (
            code == 200
            and payload.get("state") == "RUNNING"
            and int(payload.get("sleep_epoch", -1)) == initial_epochs[role] + cycle
            and payload.get("kv_memory_state") == "ACTIVE"
        )
        record(
            f"cycle {cycle} {role} RUNNING after wake",
            role_running,
            f"code={code} state={payload.get('state')} "
            f"epoch={payload.get('sleep_epoch')} "
            f"expected_epoch={initial_epochs[role] + cycle}",
        )
        identities_ok = (
            resumed_pids == pids[role]
            and len(raw_statuses) == EXPECTED_RANKS_PER_ROLE
            and all(item["state"] == "RUNNING" for item in raw_statuses)
            and identities_preserved(raw_statuses, identities)
        )
        record(
            f"cycle {cycle} {role} rank identities preserved after wake",
            identities_ok,
            f"expected={sorted(pids[role])} observed={sorted(resumed_pids)}",
        )
        running_status_ok = running_status_ok and role_running and identities_ok

    if not running_status_ok:
        raise ScenarioStepError("post-wake status validation failed")

    resumed_snapshot = memory_snapshot(pids)
    cycle_summary["after_wake_memory"] = resumed_snapshot
    if not check_running_memory(
        cycle, "after wake", resumed_snapshot, expected_pid_count
    ):
        raise ScenarioStepError("post-wake GPU memory validation failed")
    cycle_summary["post_wake_inference"] = checked_inference(
        f"cycle {cycle} post-wake inference matches baseline and golden", baseline
    )
    if not cycle_summary["post_wake_inference"]["matches_expected"]:
        cycle_summary["post_wake_failure_diag"] = capture_postwake_failure(
            cycle, baseline, cycle_summary["post_wake_inference"]
        )
        raise ScenarioStepError("post-wake inference validation failed")
    cycle_summary["completed"] = True


def build_summary(
    baseline: str,
    pids: Dict[str, Set[int]],
    identities: Dict[int, int],
    initial_backends: Dict[str, Any],
) -> Dict[str, Any]:
    failed_checks = [
        {"name": name, "detail": detail} for name, ok, detail in results if not ok
    ]
    passed = len(results) - len(failed_checks)
    return {
        "schema_version": 1,
        "scenario": f"pd_level{SLEEP_LEVEL}_sleep_wake",
        "model": MODEL,
        "cycles_requested": SLEEP_WAKE_CYCLES,
        "cycles_completed": sum(
            1 for cycle in cycle_summaries if cycle.get("completed") is True
        ),
        "success": bool(results) and not failed_checks,
        "checks": {
            "passed": passed,
            "failed": len(failed_checks),
            "total": len(results),
            "failures": failed_checks,
        },
        "expected_text": EXPECTED_TEXT,
        "baseline_text": baseline,
        "gpu_checkpoint_baseline": {
            "fallback_baseline_mib": GPU_BASELINE_MIB,
            "initial_physical_gpu_memory_mib": initial_physical_gpu_memory_mib,
            "slack_mib": MAX_CHECKPOINTED_GPU_MIB,
        },
        "roles": {
            role: {
                "gpus": sorted(_gpu_ids(ROLE_GPUS[role])),
                "control_addresses": ROLE_CONTROL_ADDRESSES[role],
                "rank_pids": sorted(pids.get(role, set())),
                "rank_starttimes": {
                    str(pid): identities[pid]
                    for pid in sorted(pids.get(role, set()))
                    if pid in identities
                },
                "initial_backends": initial_backends.get(role, []),
            }
            for role in ("decode", "prefill")
        },
        "cycles": cycle_summaries,
    }


def emit_summary(summary: Dict[str, Any]) -> None:
    encoded = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    print("\n===== PD LEVEL-3 SUMMARY =====")
    for name, ok, _ in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"{summary['checks']['passed']}/{summary['checks']['total']} passed")
    print(f"PD_LEVEL3_SUMMARY_JSON={encoded}", flush=True)
    if SUMMARY_PATH:
        summary_path = Path(SUMMARY_PATH)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(encoded + "\n", encoding="utf-8")


def main() -> int:
    global initial_physical_gpu_memory_mib
    results.clear()
    cycle_summaries.clear()
    initial_physical_gpu_memory_mib = {}
    pids: Dict[str, Set[int]] = {"decode": set(), "prefill": set()}
    identities: Dict[int, int] = {}
    initial_backends: Dict[str, Any] = {}
    initial_epochs: Dict[str, int] = {}
    sleeping_roles: Set[str] = set()
    wake_attempted_roles: Set[str] = set()
    wake_blocked_roles: Set[str] = set()
    baseline = ""
    active_cycle = 0

    try:
        for role in ("decode", "prefill"):
            code, _ = http(role, "GET", "/health")
            record(f"{role} health", code == 200, f"code={code}")

        pids, identities, initial_epochs, initial_backends = require_running_statuses()
        initial_snapshot = memory_snapshot(pids)
        initial_physical_gpu_memory_mib = dict(
            initial_snapshot["physical_gpu_memory_mib"]
        )
        check_running_memory(
            0,
            "initially",
            initial_snapshot,
            sum(len(role_pids) for role_pids in pids.values()),
        )

        code, baseline_response = infer()
        baseline = response_text(baseline_response) if code == 200 else ""
        record(
            "baseline inference matches smoke golden",
            code == 200 and baseline == EXPECTED_TEXT,
            f"code={code} text={baseline!r}",
        )

        for cycle in range(1, SLEEP_WAKE_CYCLES + 1):
            active_cycle = cycle
            run_cycle(
                cycle,
                baseline,
                pids,
                identities,
                initial_epochs,
                sleeping_roles,
                wake_attempted_roles,
                wake_blocked_roles,
            )
    except Exception as error:
        record("scenario completed without exception", False, repr(error))
    finally:
        try:
            wake_roles(
                active_cycle,
                sleeping_roles,
                wake_attempted_roles,
                wake_blocked_roles,
                {},
                cleanup=True,
            )
        except Exception as error:
            record("cleanup wake sequence", False, repr(error))

        summary = build_summary(baseline, pids, identities, initial_backends)
        emit_summary(summary)

    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
