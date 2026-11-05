"""One GPU-locked native P/D role; the coordinator releases both after HTTP.

This uses the existing launch and native_smoke entry points and the actual
CacheStore transport. Both hosts must run the same committed source/runtime.
"""

import ast
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import psutil


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def healthy(url):
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url + "/health", timeout=3) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def wait_for(predicate, process, deadline, description):
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"native server exited during {description}: {process.returncode}"
            )
        if predicate():
            return
        time.sleep(1)
    raise TimeoutError(description)


def stop_owned(process, output):
    """Signal only this invocation's launcher and its recorded descendants."""
    if process.poll() is not None:
        return process.returncode
    parent = psutil.Process(process.pid)
    children = parent.children(recursive=True)
    write_json(
        output / "owned-processes.json",
        [
            {"pid": child.pid, "created": child.create_time(), "argv": child.cmdline()}
            for child in [parent, *children]
        ],
    )
    process.terminate()
    try:
        return process.wait(timeout=240)
    except subprocess.TimeoutExpired:
        for child in children:
            if child.is_running():
                child.terminate()
        _, alive = psutil.wait_procs(children, timeout=15)
        for child in alive:
            child.kill()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=15)
        raise RuntimeError(
            "native role did not stop gracefully; scoped cleanup was required"
        )


def main():
    role = os.environ["RTP_V41_CED_ROLE"]
    encoder_replay = os.environ.get("RTP_V41_TEST_ENCODER_REPLAY", "0") == "1"
    if role not in ("prefill", "decode"):
        raise ValueError("RTP_V41_CED_ROLE must be prefill or decode")
    source = Path(os.environ["RTP_V41_TEST_SOURCE_ROOT"]).resolve()
    runtime = Path(os.environ["RTP_V41_TEST_RUNTIME"]).resolve()
    control = Path(os.environ["RTP_V41_CED_CONTROL"]).resolve()
    output = Path(os.environ["RTP_V41_CED_OUTPUT"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    control.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads((source / ".v41_snapshot.json").read_text())
    runtime_snapshot = json.loads((runtime / ".v41_snapshot.json").read_text())
    for key in ("snapshot_id", "source_sha256", "public_commit", "internal_commit"):
        if snapshot[key] != runtime_snapshot[key]:
            raise ValueError(f"source/runtime mismatch: {key}")
    checkpoint = os.environ["RTP_V41_TEST_CHECKPOINT"]
    p_address = os.environ["RTP_V41_CED_PREFILL_ADDRESS"]
    d_address = os.environ["RTP_V41_CED_DECODE_ADDRESS"]
    address = p_address if role == "prefill" else d_address
    port = int(address.rsplit(":", 1)[1])
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(runtime) + ":" + env.get("PYTHONPATH", ""),
        LOG_PATH=str(output / "logs"),
        RTP_V41_PERF_EVIDENCE_DIR=str(output / "counters"),
    )
    argv = [
        sys.executable,
        "-B",
        "-m",
        "rtp_llm.tools.v41.launch",
        role,
        "--checkpoint-path",
        checkpoint,
        "--prefill-address",
        p_address,
        "--decode-address",
        d_address,
        "--start-port",
        str(port),
        "--ced-mode",
        "bounded",
        "--token-map-cache",
        str(output / "token-map.json"),
    ]
    if encoder_replay:
        argv.append("--encoder-swa-replay")
    deadline = time.monotonic() + 6300
    with (output / "process.log").open("x") as log:
        process = subprocess.Popen(
            argv, cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        write_json(
            output / "launch.json",
            {
                "pid": process.pid,
                "argv": argv,
                "cwd": str(runtime),
                "role": role,
                "snapshot_id": snapshot["snapshot_id"],
                "ced": True,
                "bounded_replay": True,
            },
        )
        try:
            local_urls = [
                f"http://127.0.0.1:{port + rank * 9}"
                for rank in range(8 if role == "decode" else 1)
            ]
            wait_for(
                lambda: all(map(healthy, local_urls)),
                process,
                min(deadline, time.monotonic() + 2700),
                "all native HTTP listeners ready",
            )
            write_json(output / "ready.json", {"role": role, "listeners": local_urls})
            if role == "prefill":
                host, base = d_address.rsplit(":", 1)
                urls = [f"http://{host}:{int(base) + rank * 9}" for rank in range(8)]
                wait_for(
                    lambda: all(map(healthy, urls)),
                    process,
                    deadline,
                    "all remote D owner listeners ready",
                )
                for repeats, requests in (
                    ((1024, 9),) if encoder_replay else ((128, 1), (256, 8))
                ):
                    subprocess.run(
                        [
                            sys.executable,
                            "-B",
                            "-m",
                            "rtp_llm.tools.v41.native_smoke",
                            "--url",
                            "http://" + p_address,
                            "--prompt",
                            "hello " * repeats,
                            "--expected-input-tokens",
                            str(repeats + 1),
                            "--requests",
                            str(requests),
                            "--max-new-tokens",
                            "4",
                            "--timeout",
                            "1200",
                            "--output",
                            str(output / f"http-{repeats + 1}"),
                            *(["--reuse-cache"] if encoder_replay else []),
                        ],
                        cwd=runtime,
                        env=env,
                        check=True,
                        timeout=deadline - time.monotonic(),
                    )
                if encoder_replay:
                    http = json.loads((output / "http-1025/summary.json").read_text())
                    reused = [
                        request["aux_info"].get("prefill_total_reuse_len", 0)
                        for request in http["requests"]
                    ]
                    if not all(length > 0 for length in reused[1:]):
                        raise AssertionError(
                            f"encoder recovery did not reuse Global prefixes: {reused}"
                        )
                    write_json(
                        output / "encoder-prefix-reuse.json", {"reused_lengths": reused}
                    )
                write_json(
                    output / "requests-done.json",
                    {"requests": 9, "outputs_per_request": 4},
                )
            wait_for(
                lambda: (control / "stop").exists(),
                process,
                deadline,
                "coordinator finished collecting both native roles",
            )
        finally:
            status = stop_owned(process, output)
            write_json(output / "server-exit.json", {"exit_code": status})
    if status != 0:
        raise RuntimeError(f"native server failed: {status}")
    logs = [output / "process.log", *(output / "logs").rglob("*.log")]
    text = "\n".join(path.read_text(errors="replace") for path in logs)
    if role == "prefill":
        plans = [
            ast.literal_eval(item)
            for item in re.findall(r"PPU CED compacted (\{[^\n]+\})", text)
        ]
        if not any(
            p["original_rows"] == (1025 if encoder_replay else 257)
            and p["query_rows"] == 128
            and p["first_decoder_layer"] == 21
            and p["cp_size"] == 8
            for p in plans
        ):
            raise AssertionError("missing real L21 CP8 compaction evidence")
        evidence = {"ced_plans": plans}
        if encoder_replay:
            rebuilt = re.findall(
                r"PPU encoder SWA rebuilt cp=(\d+)/8 prefix=\[(\d+)\]", text
            )
            ranks = sorted({int(rank) for rank, prefix in rebuilt if int(prefix) > 0})
            if ranks != list(range(8)):
                raise AssertionError(
                    f"missing encoder recovery evidence on CP ranks: {ranks}"
                )
            evidence["encoder_recovery_cp_ranks"] = ranks
    else:
        owners = sorted(set(map(int, re.findall(r"native DECODE dp=(\d+)", text))))
        if owners != list(range(8)) or "native PREFILL" in text:
            raise AssertionError(
                f"D must continue on every owner without prefill: {owners}"
            )
        evidence = {"decode_owners": owners, "prefill_executed": False}
    write_json(
        output / "summary.json",
        {
            "native_role_ok": True,
            "role": role,
            "snapshot_id": snapshot["snapshot_id"],
            "transport": "native TCP CacheStore",
            "evidence": evidence,
            "numerical_alignment": "deferred",
            "nv_alignment": "deferred",
        },
    )


if __name__ == "__main__":
    main()
