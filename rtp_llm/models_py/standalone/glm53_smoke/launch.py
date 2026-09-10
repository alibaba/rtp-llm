"""Guard a foreground smoke command and release only its own descendants.

Example: python -m rtp_llm.models_py.standalone.glm53_smoke.launch
  --run-dir /dataN/user/run -- python -m torch.distributed.run ...
"""

import argparse
import csv
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import uuid


def query(kind, fields):
    result = subprocess.run(
        ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return [[x.strip() for x in row] for row in csv.reader(io.StringIO(result.stdout))]


def gpu_sample():
    rows = query("gpu", "index,uuid,memory.used")
    if len(rows) != 8 or {int(x[0]) for x in rows} != set(range(8)):
        raise RuntimeError("the smoke requires all eight local GPUs")
    return dict(
        timestamp=time.time(),
        gpus=[dict(index=int(i), uuid=u, memory_mib=int(m)) for i, u, m in rows],
    )


def process_identity(pid):
    try:
        # comm may contain spaces and parentheses. Fields after its final ')'
        # start with field 3 (state), and field 22 is the start time.
        tail = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return dict(pid=int(pid), state=tail[0], start_ticks=tail[19])
    except (FileNotFoundError, ProcessLookupError):
        return None


def owned_processes(marker):
    result = []
    token = f"GLM53_SMOKE_RUN_ID={marker}".encode()
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            if token not in (path / "environ").read_bytes().split(b"\0"):
                continue
            identity = process_identity(path.name)
            if identity and identity["state"] != "Z":
                result.append(identity)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return result


def signal_verified(process, signum):
    current = process_identity(process["pid"])
    if (
        current
        and current["start_ticks"] == process["start_ticks"]
        and current["state"] != "Z"
    ):
        try:
            os.kill(process["pid"], signum)
        except ProcessLookupError:
            pass


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--budget-gib", type=float, default=180)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a foreground smoke command is required after --")
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    # Exclusive ownership record prevents accidentally reusing a live run.
    marker = str(uuid.uuid4())
    with (run / "launch.json").open("x") as stream:
        json.dump(
            dict(
                host=socket.gethostname(),
                marker=marker,
                command=command,
                started=time.time(),
                launcher=process_identity(os.getpid()),
            ),
            stream,
            indent=2,
        )
    with (run / "gate.jsonl").open("x") as stream:
        gate_samples = []
        while True:
            sample = gpu_sample()
            gate_samples.append(sample["timestamp"])
            stream.write(json.dumps(sample) + "\n")
            stream.flush()
            if any(x["memory_mib"] >= 1024 for x in sample["gpus"]):
                raise RuntimeError("GPU gate failed; select another idle host")
            if gate_samples[-1] - gate_samples[0] >= 5:
                break
            time.sleep(1)
    env = dict(
        os.environ,
        GLM53_SMOKE_RUN_ID=marker,
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
        TORCH_CUDA_ARCH_LIST="10.3",
    )
    interrupted = False

    def interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    observed = {}
    samples = []
    child = None
    exit_code = 1
    try:
        with (run / "process.log").open("x") as log, (run / "gpu.jsonl").open(
            "x"
        ) as monitor:
            child = subprocess.Popen(
                command,
                env=env,
                start_new_session=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            launch_time = time.monotonic()
            while True:
                owned = owned_processes(marker)
                owned_pids = {x["pid"] for x in owned}
                observed.update({x["pid"]: x for x in owned})
                sample = gpu_sample()
                compute = query("compute-apps", "gpu_uuid,pid,used_memory")
                sample["owned"] = owned
                sample["compute"] = [
                    dict(
                        uuid=u,
                        pid=int(p),
                        memory_mib=int(m),
                        owned=int(p) in owned_pids,
                    )
                    for u, p, m in compute
                ]
                samples.append(sample)
                monitor.write(json.dumps(sample) + "\n")
                monitor.flush()
                peak = max(
                    (
                        sum(
                            p["memory_mib"]
                            for p in sample["compute"]
                            if p["uuid"] == g["uuid"] and p["owned"]
                        )
                        for g in sample["gpus"]
                    ),
                    default=0,
                )
                if peak > args.budget_gib * 1024:
                    raise RuntimeError(
                        f"owned GPU allocation exceeds {args.budget_gib} GiB"
                    )
                if interrupted:
                    raise InterruptedError("smoke cancelled")
                if time.monotonic() - launch_time > args.timeout_seconds:
                    raise TimeoutError("smoke exceeded its bounded runtime")
                if child.poll() is not None:
                    exit_code = child.returncode
                    break
                time.sleep(0.5)
    finally:
        observed.update({x["pid"]: x for x in owned_processes(marker)})
        for proc in observed.values():
            signal_verified(proc, signal.SIGTERM)
        deadline = time.monotonic() + 5
        while owned_processes(marker) and time.monotonic() < deadline:
            time.sleep(0.5)
        for proc in owned_processes(marker):
            signal_verified(proc, signal.SIGKILL)
        evidence = []
        while True:
            evidence.append(
                dict(
                    timestamp=time.time(),
                    remaining=owned_processes(marker),
                    gpu=gpu_sample(),
                )
            )
            if evidence[-1]["timestamp"] - evidence[0]["timestamp"] >= 5:
                break
            time.sleep(1)
        if child is not None:
            child.wait(timeout=10)
        (run / "cleanup.json").write_text(
            json.dumps(
                dict(
                    observed=list(observed.values()),
                    samples=evidence,
                    stopped=not evidence[-1]["remaining"],
                ),
                indent=2,
            )
        )
        if evidence[-1]["remaining"]:
            raise RuntimeError("task-owned workers remain; see cleanup.json")
    contaminated = []
    for path in run.glob("result_rank*.json"):
        window = json.loads(path.read_text())["timing_window_unix"]
        contaminated += [
            s
            for s in samples
            if window[0] - 0.5 <= s["timestamp"] <= window[1] + 0.5
            and any(not p["owned"] for p in s["compute"])
        ]
    (run / "isolation.json").write_text(
        json.dumps(
            dict(clean=not contaminated, contaminated_samples=contaminated), indent=2
        )
    )
    if contaminated:
        raise RuntimeError(
            "foreign GPU processes overlapped timing; repeat on an idle host"
        )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
