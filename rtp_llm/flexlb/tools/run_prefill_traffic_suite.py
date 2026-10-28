#!/usr/bin/env python3
"""Replay continuous, seeded traffic through a real Master and controlled Mock workers."""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(path):
    catalog = json.loads(path.read_text())
    if catalog.get("schema_version") != 1:
        raise ValueError("unsupported case schema")
    if not catalog.get("cases"):
        raise ValueError("empty case catalog")
    names = set()
    for case in catalog["cases"]:
        name = case["name"]
        if (
            name in names
            or not name
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in name)
        ):
            raise ValueError(f"invalid/duplicate case name: {name}")
        names.add(name)
        for field in (
            "cold_fraction",
            "long_fraction",
            "session_fraction",
            "hot_fraction",
        ):
            if not 0 <= case.get(field, 0) <= 1:
                raise ValueError(f"invalid {field}: {name}")
        if case.get("arrival", "poisson") not in ("poisson", "uniform"):
            raise ValueError("unknown arrival distribution")
        prediction_scale = case.get("prediction_scale", 1.0)
        if (
            type(prediction_scale) not in (int, float)
            or not math.isfinite(prediction_scale)
            or prediction_scale <= 0
        ):
            raise ValueError(f"invalid prediction_scale: {name}")
        cache_blocks = case.get("prefill_cache_blocks", 100000)
        if type(cache_blocks) is not int or cache_blocks <= 0:
            raise ValueError(f"invalid prefill_cache_blocks: {name}")
        stage_names = set()
        if not case.get("phases"):
            raise ValueError("case has no phases")
        for phase in case["phases"]:
            if phase["name"] in stage_names:
                raise ValueError("phase names must be unique")
            stage_names.add(phase["name"])
            if (
                not math.isfinite(phase["seconds"])
                or phase["seconds"] <= 0
                or not math.isfinite(phase["load"])
                or phase["load"] <= 0
            ):
                raise ValueError("phase duration and load must be positive")
            if (
                type(phase.get("workers", 3)) is not int
                or not 2 <= phase.get("workers", 3) <= 5
            ):
                raise ValueError("worker count must be 2..5")
            if (
                not math.isfinite(phase.get("burst_period_s", 1))
                or not math.isfinite(phase.get("ramp_end_load", 1))
                or phase.get("burst_period_s", 1) <= 0
                or phase.get("ramp_end_load", 1) <= 0
            ):
                raise ValueError("invalid burst period or ramp load")
            if (
                type(phase.get("burst_requests", 0)) is not int
                or phase.get("burst_requests", 0) < 0
            ):
                raise ValueError("burst request count must be a nonnegative integer")
    return catalog["cases"]


def make_trace(case, seed):
    """Generate arrivals and complete request inputs before touching an RPC or clock."""
    rng = random.Random(seed)
    rid = 30000

    def shape(index, cold, stage, offset):
        nonlocal rid
        rid += 1
        length = 65536 if index < round(case["long_fraction"] * 100) else 4096
        base = (rid + 100000 if cold else index + 1) * 1000
        keys = [base + block for block in range(length // 1024 - 1)] + [rid * 100000]
        session = ""
        # A separate deterministic permutation avoids correlating ID presence with length.
        if not cold and (index * 37) % 100 < case["session_fraction"] * 100:
            session = f"conversation-{index}"
        elif cold and case.get("session_for_cold", False):
            session = f"cold-conversation-{rid}"
        return dict(
            stage=stage,
            request_id=rid,
            session=session,
            input_tokens=length,
            keys=keys,
            offset_s=offset,
            cold=cold,
        )

    warmup = [shape(i, False, "warmup", 0) for i in range(101)]
    # Identical session with unrelated cold keys tests session affinity independent of KV hits.
    probes = [shape(100, True, "affinity", 0) for _ in range(2)]
    for probe in probes:
        probe["session"] = "cold-affinity-probe"
    # This also establishes that a configured BATCH path can deliver a real grouped batch.
    calibration = [shape(100, False, "calibration", 0) for _ in range(16)]
    cold = case["cold_fraction"]
    hot = case.get("hot_fraction", 0)
    # The hot conversation is short; account for its changed input mix in nominal demand.
    realized_long_fraction = round(case["long_fraction"] * 100) / 100
    mean_input = (1 - hot) * (
        realized_long_fraction * 65536 + (1 - realized_long_fraction) * 4096
    ) + hot * 4096
    mean_compute = cold * mean_input + (1 - cold) * 1024
    trace, phases, offset = [], [], 0.0
    for phase in case["phases"]:
        phase = dict(phase, offset_s=offset, nominal_mean_compute_tokens=mean_compute)
        seconds = phase["seconds"]
        initial = phase["load"] * 3 * 70000 / mean_compute
        final = phase.get("ramp_end_load", phase["load"]) * 3 * 70000 / mean_compute
        phase["nominal_qps_start"] = initial
        phase["nominal_qps_end"] = final
        # Invert integrated intensity for a Poisson (or uniform) linear rate ramp.
        mass = 0.0
        while True:
            mass += (
                rng.expovariate(1) if case.get("arrival", "poisson") == "poisson" else 1
            )
            slope = (final - initial) / seconds
            discriminant = initial * initial + 2 * slope * mass
            if discriminant <= 0:
                break
            elapsed = 2 * mass / (initial + math.sqrt(discriminant))
            if elapsed >= seconds:
                break
            index = 100 if rng.random() < hot else rng.randrange(100)
            trace.append(
                shape(index, rng.random() < cold, phase["name"], offset + elapsed)
            )
        burst_at = 0.0
        while phase.get("burst_requests", 0) and burst_at < seconds:
            for _ in range(phase["burst_requests"]):
                trace.append(
                    shape(rng.randrange(100), True, phase["name"], offset + burst_at)
                )
            if "burst_period_s" not in phase:
                break
            burst_at += phase["burst_period_s"]
        phases.append(phase)
        offset += seconds
    trace.sort(key=lambda row: (row["offset_s"], row["request_id"]))
    return warmup, probes, calibration, trace, phases


def annotate_slo(audit, rows, slo_ms):
    """Count deadline goodput over every arrival, not just the successful cohort."""
    audit["ttft_slo_ms"] = slo_ms
    for phase in audit["stages"]:
        samples = [row for row in rows if row["stage"] == phase["stage"]]
        within = sum(
            row["success"] and row["ttft_s"] * 1000 <= slo_ms for row in samples
        )
        phase["slo_successes"] = within
        phase["slo_goodput_fraction"] = within / len(samples)
        phase["slo_success_fraction"] = within / phase["completed"]
        if (
            phase.get("allow_rejections") is False
            and phase["ttft_p99_s"] * 1000 > slo_ms
        ):
            raise ValueError(f"low-load phase {phase['stage']} misses TTFT target")
        if phase.get("require_tail_zero_rejections", False):
            tail = [
                row
                for row in samples
                if row["planned_s"] - phase["start_s"] >= phase["settle_seconds"]
            ]
            timely = sum(
                row["success"] and row["ttft_s"] * 1000 <= slo_ms for row in tail
            )
            phase["recovery_tail_slo_goodput_fraction"] = (
                timely / len(tail) if tail else 0
            )
            if not tail or timely / len(tail) < 0.99:
                raise ValueError(f"recovery tail {phase['stage']} misses TTFT target")


def master_config(dispatch, learning, slo_ms=None, prediction_scale=1.0):
    if prediction_scale != 1.0 and (learning or slo_ms is not None):
        raise ValueError("prediction_scale requires explicit FORMULA configuration")
    if slo_ms is not None:
        if slo_ms <= 0:
            raise ValueError("slo_ms must be positive")
        config = {"ttftSloMs": slo_ms}
        if dispatch != "NON_BATCH":
            config["dispatcher"] = {"type": dispatch}
        return config
    decision = {"type": "SINGLE"}
    if dispatch == "BATCH":
        decision = {
            "type": "FIXED_WINDOW",
            "maxRequests": 32,
            "maxCollectionWaitMs": 20,
        }
    prefill = {"tokenBalance": {}}
    if not learning:
        prefill["executionTimeEstimator"] = {
            "type": "FORMULA",
            "expression": "sum(computeTokens) / 70"
            + (f" * {prediction_scale}" if prediction_scale != 1.0 else ""),
        }
    return {
        "schemaVersion": 3,
        "router": {"roles": {"prefill": prefill}},
        "scheduler": {
            "type": "QUEUE",
            "ordering": {"type": "FIFO"},
            "decision": decision,
            "queueTimeoutMs": 6000,
            "prefillQueueShedding": {
                "expectedWaitThresholdMs": 3000,
                "maxWaitMs": 6000,
            },
        },
        "dispatcher": {
            "type": dispatch,
            "maxInflightPerPrefillWorker": 32 if dispatch == "NON_BATCH" else 2,
        },
        "requestLifecycle": {
            "request": {"timeoutMs": 30000},
            "decision": {"lifetime": 2.0},
        },
        "workerRegistry": {
            "health": {
                "statusPollIntervalMs": 20,
                "statusRpcTimeoutMs": 200,
                "statusStaleAfterMs": 500,
                "cleanupIntervalMs": 200,
            }
        },
    }


def run_case(args, case):
    from prefill_traffic_audit import audit_run, validate_environment, validate_trace
    from prefill_traffic_client import (
        AsyncQueryClient,
        start_engine_stream,
        suspend_cyclic_gc,
    )
    from prefill_worker_jitter import JitterController, audit_execution

    out = args.out_dir.resolve()
    os.environ["FLEXLB_FT_MASTER_JAR"] = str(args.jar.resolve())
    os.environ.pop("FLEXLB_FT_MASTER_CONFIG_FILE", None)
    sys.path[:0] = [str(args.runner), str(args.runner / "src")]
    from runtime import paths

    paths.MOCK_JAR = args.mock_jar.resolve()
    from adapters import proto_utils
    from adapters.engine_ops import EngineOps
    from adapters.environment import EnvManager
    from adapters.environment_config import EnvSpec
    from runtime.network import http_post_json

    proto_utils.PROTO_DIR = HERE.parents[1] / "cpp/model_rpc/proto"
    config = master_config(
        args.dispatch,
        case.get("learning", False),
        args.slo_ms,
        case.get("prediction_scale", 1.0),
    )
    mock_config = master_config(args.dispatch, False)
    del mock_config["router"]["roles"]["prefill"]["tokenBalance"]
    del mock_config["scheduler"]["prefillQueueShedding"]
    mock_config["router"]["roles"]["prefill"]["executionTimeEstimator"] = {
        "type": "FORMULA",
        "expression": "sum(computeTokens) / 70",
    }
    spec = EnvSpec(
        label="traffic",
        n_prefill=3,
        n_decode=6,
        discovery="discovery_file",
        master_profile=(
            "batch-window" if args.dispatch == "BATCH" else "single-nonbatch"
        ),
        raw_config=json.dumps(mock_config),
        master_jvm_heap="2g",
        mock_heap="2g",
        prefill_cache_blocks=case.get("prefill_cache_blocks", 100000),
    )
    spec.perf["prefill_expression"] = "sum(computeTokens) / 70"
    write_json(out / "config.json", config)
    os.environ["FLEXLB_FT_MASTER_CONFIG_FILE"] = str(out / "config.json")
    warmup, probes, calibration, trace, phases = make_trace(case, args.seed)
    write_json(
        out / "trace.json",
        {
            "warmup": warmup,
            "affinity": probes,
            "calibration": calibration,
            "traffic": trace,
        },
    )
    manifest = {
        "case": case,
        "dispatch": args.dispatch,
        "jitter": args.jitter,
        "seed": args.seed,
        "master_sha256": sha256(args.jar),
        "mock_sha256": sha256(args.mock_jar),
        "trace_sha256": sha256(out / "trace.json"),
        "runner": str(args.runner.resolve()),
        "limitations": "Real Master; synthetic linear Prefill and Mock Decode. No GPU/model/production SLO claim.",
    }
    source_dir = out / "source"
    source_dir.mkdir()
    for name in (
        Path(__file__).name,
        "prefill_worker_jitter.py",
        "prefill_traffic_audit.py",
        "prefill_traffic_client.py",
    ):
        (source_dir / name).write_bytes((HERE / name).read_bytes())
    manifest["source_sha256"] = {p.name: sha256(p) for p in source_dir.iterdir()}
    manifest["schedule_proto_sha256"] = sha256(
        proto_utils.PROTO_DIR / "flexlb_schedule_service.proto"
    )
    write_json(out / "manifest.json", manifest)
    manager = EnvManager(out / "environment")
    ops, jitter = None, None
    rows, topology = [], []
    stages = []

    def wait_topology(expected):
        until = time.monotonic() + 15
        while time.monotonic() < until:
            status, data = http_post_json(env.master_http("/rtp_llm/master/info"), {})
            summary = (data or {}).get("worker_summary", {}).get("PREFILL", {})
            if (
                status == 200
                and summary.get("alive") == expected
                and summary.get("discovered") == expected
            ):
                topology.append(
                    {
                        "epoch_ms": time.time() * 1000,
                        "expected": expected,
                        "master": data,
                    }
                )
                return
            time.sleep(0.05)
        raise AssertionError(f"topology failed to converge to {expected}: {data}")

    def resize(expected):
        current = [e for e in ops.snapshot()["engines"] if e["role"] == "prefill"]
        while len(current) < expected:
            code, body = ops.add_engine("prefill")
            assert code == 200, body
            topology.append({"epoch_ms": time.time() * 1000, "add": body})
            current = [e for e in ops.snapshot()["engines"] if e["role"] == "prefill"]
        while len(current) > expected:
            # Drain an old warm owner, so shrinking exercises lost affinity and cold replacements.
            worker = min(current, key=lambda e: e["name"])
            discovery = json.loads(env.discovery_file.read_text())
            address = worker["http_addr"]
            assert any(address in addresses for addresses in discovery.values())
            for domain, addresses in discovery.items():
                discovery[domain] = [value for value in addresses if value != address]
            temporary = env.discovery_file.with_suffix(".withdraw.tmp")
            write_json(temporary, discovery)
            temporary.replace(env.discovery_file)
            wait_topology(len(current) - 1)
            code, body = ops.remove_engine(
                worker["name"],
                mode="graceful",
                drain_timeout_ms=max(15000, 2 * (args.slo_ms or 3000) + 5000),
            )
            assert code == 200 and body.get("drained") is True, body
            topology.append({"epoch_ms": time.time() * 1000, "remove": body})
            current = [e for e in ops.snapshot()["engines"] if e["role"] == "prefill"]
        wait_topology(expected)

    try:
        env = manager.ensure(spec)
        ops = EngineOps(
            "127.0.0.1",
            env.master_http_port,
            env.mock_http_port,
            env.master_management_port,
        )
        transport_timeout_s = max(20.0, 2 * (args.slo_ms or 3000) / 1000 + 5.0)
        with suspend_cyclic_gc() as controls, AsyncQueryClient(
            ops,
            schedule_timeout_s=transport_timeout_s,
            stream_timeout_s=transport_timeout_s,
            request_timeout_ms=max(30000, 4 * args.slo_ms) if args.slo_ms else None,
            stream_starter=lambda response, rid, generated: start_engine_stream(
                ops, response, rid, generated, transport_timeout_s
            ),
        ) as client:
            wait_topology(3)
            for item in warmup + probes:
                row = client.submit(item, time.monotonic(), time.time() * 1000).result()
                rows.append(row)
                assert row["success"], row
            assert (
                rows[-1]["worker"] == rows[-2]["worker"]
            ), "cold-key session owner changed without pressure"
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as changes:
                futures = [
                    client.submit(item, time.monotonic(), time.time() * 1000)
                    for item in calibration
                ]
                rows.extend(future.result() for future in futures)
                assert all(
                    row["success"] for row in rows
                ), "calibration delivery failed"
                jitter = JitterController(
                    ops, mode=args.jitter, seed=args.seed, period_s=1.0
                )
                controls.callback(jitter.stop)
                jitter.start()
                start = time.monotonic() + 0.1
                epoch = time.time() * 1000 + (start - time.monotonic()) * 1000
                for phase in phases:
                    stages.append(
                        dict(
                            phase,
                            stage=phase["name"],
                            start_s=start + phase["offset_s"],
                            start_epoch_ms=epoch + phase["offset_s"] * 1000,
                            end_epoch_ms=epoch
                            + (phase["offset_s"] + phase["seconds"]) * 1000,
                            worker_count=phase.get("workers", 3),
                        )
                    )
                timeline = [(phase["offset_s"], 0, phase) for phase in phases]
                timeline.extend((item["offset_s"], 1, item) for item in trace)
                timeline.sort(key=lambda entry: (entry[0], entry[1]))
                futures, mutations = [], []
                workers = 3
                for offset, kind, item in timeline:
                    planned = start + offset
                    remaining = planned - time.monotonic()
                    if remaining > 0:
                        time.sleep(remaining)
                    if kind == 0:
                        if item.get("workers", workers) != workers:
                            workers = item["workers"]
                            mutations.append(changes.submit(resize, workers))
                        print(
                            json.dumps(
                                {
                                    "phase": item["name"],
                                    "workers": workers,
                                    "load": item["load"],
                                }
                            ),
                            flush=True,
                        )
                    else:
                        futures.append(
                            client.submit(item, planned, epoch + offset * 1000)
                        )
                time.sleep(
                    max(0, start + sum(p["seconds"] for p in phases) - time.monotonic())
                )
                rows.extend(future.result() for future in futures)
                for mutation in mutations:
                    mutation.result()
                for stage in stages:
                    cohort = [row for row in rows if row["stage"] == stage["stage"]]
                    stage["drained_s"] = max(
                        [stage["start_s"] + stage["seconds"]]
                        + [row["ended_s"] for row in cohort]
                    )
                    stage["issued_end_s"] = stage["start_s"] + stage["seconds"]
    finally:
        try:
            if jitter is not None:
                try:
                    jitter.stop()
                finally:
                    write_json(out / "worker-performance.json", jitter.records)
        finally:
            try:
                if ops is not None:
                    try:
                        write_json(out / "final-snapshot.json", ops.snapshot())
                    finally:
                        ops.close()
            finally:
                manager.teardown()
                write_json(out / "requests.json", rows)
                write_json(out / "stages.json", stages)
                write_json(out / "topology.json", topology)
    events = []
    for path in (out / "environment").rglob("engine_events.jsonl"):
        events.extend(json.loads(line) for line in path.read_text().splitlines())
    performance = audit_execution(events, jitter.records, mode=args.jitter)
    write_json(out / "performance-audit.json", performance)
    validate_trace(rows, json.loads((out / "trace.json").read_text()), stages)
    validate_environment(
        stages, topology, json.loads((out / "final-snapshot.json").read_text())
    )
    audit = audit_run(
        rows,
        stages,
        events,
        dispatch=args.dispatch,
        hard_wait_s=2 * args.slo_ms / 1000 if args.slo_ms else 6.0,
    )
    if args.slo_ms is not None:
        annotate_slo(audit, rows, args.slo_ms)
    audit["performance"] = performance
    audit["trace_sha256"] = manifest["trace_sha256"]
    audit["status"] = "PASS"
    write_json(out / "report.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, default=HERE / "prefill_traffic_cases.json"
    )
    parser.add_argument("--suite", choices=("quick", "acceptance"), default="quick")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--jar", type=Path)
    parser.add_argument("--mock-jar", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--dispatchers",
        nargs="+",
        choices=("NON_BATCH", "BATCH"),
        default=["NON_BATCH", "BATCH"],
    )
    parser.add_argument(
        "--jitters", nargs="+", choices=("stable", "independent", "correlated")
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--slo-ms",
        type=int,
        help="Use only this TTFT SLO and transport mode; learn all worker service times",
    )
    parser.add_argument("--jobs", type=int, choices=(1, 2, 4, 8, 16), default=1)
    parser.add_argument("--run-case", help=argparse.SUPPRESS)
    parser.add_argument(
        "--dispatch", choices=("NON_BATCH", "BATCH"), help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--jitter",
        choices=("stable", "independent", "correlated"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.slo_ms is not None and args.slo_ms <= 0:
        parser.error("--slo-ms must be positive")
    cases = load_cases(args.cases)
    if args.list:
        print(json.dumps(cases, indent=2))
        return
    for field in ("runner", "jar", "mock_jar", "out_dir"):
        if getattr(args, field) is None:
            parser.error(f"--{field.replace('_', '-')} is required")
    if args.run_case:
        # Creation is the ownership boundary: an existing run must stay untouched.
        args.out_dir.mkdir(parents=True, exist_ok=False)
        try:
            run_case(args, next(c for c in cases if c["name"] == args.run_case))
        except Exception as error:
            if args.out_dir.is_dir():
                write_json(
                    args.out_dir / "failure.json",
                    {
                        "status": "FAIL",
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    },
                )
            raise
        return
    selected = (
        [c for c in cases if c["name"] in args.case]
        if args.case
        else [c for c in cases if args.suite in c["suites"]]
    )
    if not selected or set(args.case) - {c["name"] for c in selected}:
        parser.error("empty or unknown case selection")
    jobs = []
    for case in selected:
        seeds = args.seeds or ([1] if args.suite == "quick" else case["seeds"])
        jitters = args.jitters or (
            ["stable", "independent"] if args.suite == "quick" else case["jitters"]
        )
        jobs.extend(
            (case["name"], dispatch, jitter, seed)
            for seed in seeds
            for dispatch in args.dispatchers
            for jitter in jitters
        )

    if len(jobs) != len(set(jobs)):
        parser.error("duplicate case/dispatch/jitter/seed combinations")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.out_dir / "cases.json", {"schema_version": 1, "cases": selected})

    def execute(slot_jobs, slot):
        completed = []
        for name, dispatch, jitter, seed in slot_jobs:
            label = f"{name}-{dispatch.lower()}-{jitter}-s{seed}"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--cases",
                str((args.out_dir / "cases.json").resolve()),
                "--runner",
                str(args.runner.resolve()),
                "--jar",
                str(args.jar.resolve()),
                "--mock-jar",
                str(args.mock_jar.resolve()),
                "--out-dir",
                str((args.out_dir / label).resolve()),
                "--run-case",
                name,
                "--dispatch",
                dispatch,
                "--jitter",
                jitter,
                "--seed",
                str(seed),
            ]
            if args.slo_ms is not None:
                cmd.extend(["--slo-ms", str(args.slo_ms)])
            env = dict(
                os.environ,
                FLEXLB_FT_MASTER_HTTP_PORT=str(17480 + slot * 20),
                FLEXLB_FT_MASTER_MANAGEMENT_PORT=str(17481 + slot * 20),
                FLEXLB_FT_MOCK_BASE_GRPC_PORT=str(
                    12000 + slot * 1000 if slot < 4 else 24000 + (slot - 4) * 1000
                ),
            )
            with (args.out_dir / f"{label}.log").open("w") as log:
                process = subprocess.run(
                    cmd, env=env, stdout=log, stderr=subprocess.STDOUT
                )
            result = dict(
                case=name,
                dispatch=dispatch,
                jitter=jitter,
                seed=seed,
                directory=label,
                exit_code=process.returncode,
            )
            completed.append(result)
            write_json(args.out_dir / f"results-slot-{slot}.json", completed)
            print(json.dumps(result), flush=True)
        return completed

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(execute, jobs[slot :: args.jobs], slot)
            for slot in range(args.jobs)
        ]
        results = [row for future in futures for row in future.result()]
    # Paired controls must consume precisely the same arrivals, keys, lengths and session IDs.
    traces = {}
    for result in results:
        path = args.out_dir / result["directory"] / "manifest.json"
        if path.exists():
            manifest = json.loads(path.read_text())
            key = (result["case"], result["seed"])
            previous = traces.setdefault(key, manifest["trace_sha256"])
            assert previous == manifest["trace_sha256"], "paired traffic inputs differ"
    write_json(args.out_dir / "results.json", results)
    if any(row["exit_code"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
